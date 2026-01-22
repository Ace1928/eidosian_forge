import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
class LocalFileSystem(AbstractFileSystem):
    """Interface to files on local storage

    Parameters
    ----------
    auto_mkdir: bool
        Whether, when opening a file, the directory containing it should
        be created (if it doesn't already exist). This is assumed by pyarrow
        code.
    """
    root_marker = '/'
    protocol = ('file', 'local')
    local_file = True

    def __init__(self, auto_mkdir=False, **kwargs):
        super().__init__(**kwargs)
        self.auto_mkdir = auto_mkdir

    @property
    def fsid(self):
        return 'local'

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if self.exists(path):
            raise FileExistsError(path)
        if create_parents:
            self.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path, **kwargs)

    def makedirs(self, path, exist_ok=False):
        path = self._strip_protocol(path)
        os.makedirs(path, exist_ok=exist_ok)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        os.rmdir(path)

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        info = self.info(path)
        if info['type'] == 'directory':
            with os.scandir(path) as it:
                infos = [self.info(f) for f in it]
        else:
            infos = [info]
        if not detail:
            return [i['name'] for i in infos]
        return infos

    def info(self, path, **kwargs):
        if isinstance(path, os.DirEntry):
            out = path.stat(follow_symlinks=False)
            link = path.is_symlink()
            if path.is_dir(follow_symlinks=False):
                t = 'directory'
            elif path.is_file(follow_symlinks=False):
                t = 'file'
            else:
                t = 'other'
            path = self._strip_protocol(path.path)
        else:
            path = self._strip_protocol(path)
            out = os.stat(path, follow_symlinks=False)
            link = stat.S_ISLNK(out.st_mode)
            if link:
                out = os.stat(path, follow_symlinks=True)
            if stat.S_ISDIR(out.st_mode):
                t = 'directory'
            elif stat.S_ISREG(out.st_mode):
                t = 'file'
            else:
                t = 'other'
        result = {'name': path, 'size': out.st_size, 'type': t, 'created': out.st_ctime, 'islink': link}
        for field in ['mode', 'uid', 'gid', 'mtime', 'ino', 'nlink']:
            result[field] = getattr(out, f'st_{field}')
        if result['islink']:
            result['destination'] = os.readlink(path)
            try:
                out2 = os.stat(path, follow_symlinks=True)
                result['size'] = out2.st_size
            except OSError:
                result['size'] = 0
        return result

    def lexists(self, path, **kwargs):
        return osp.lexists(path)

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip('/')
        path2 = self._strip_protocol(path2).rstrip('/')
        if self.auto_mkdir:
            self.makedirs(self._parent(path2), exist_ok=True)
        if self.isfile(path1):
            shutil.copyfile(path1, path2)
        elif self.isdir(path1):
            self.mkdirs(path2, exist_ok=True)
        else:
            raise FileNotFoundError(path1)

    def get_file(self, path1, path2, callback=None, **kwargs):
        if isfilelike(path2):
            with open(path1, 'rb') as f:
                shutil.copyfileobj(f, path2)
        else:
            return self.cp_file(path1, path2, **kwargs)

    def put_file(self, path1, path2, callback=None, **kwargs):
        return self.cp_file(path1, path2, **kwargs)

    def mv_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip('/')
        path2 = self._strip_protocol(path2).rstrip('/')
        shutil.move(path1, path2)

    def link(self, src, dst, **kwargs):
        src = self._strip_protocol(src)
        dst = self._strip_protocol(dst)
        os.link(src, dst, **kwargs)

    def symlink(self, src, dst, **kwargs):
        src = self._strip_protocol(src)
        dst = self._strip_protocol(dst)
        os.symlink(src, dst, **kwargs)

    def islink(self, path) -> bool:
        return os.path.islink(self._strip_protocol(path))

    def rm_file(self, path):
        os.remove(self._strip_protocol(path))

    def rm(self, path, recursive=False, maxdepth=None):
        if not isinstance(path, list):
            path = [path]
        for p in path:
            p = self._strip_protocol(p).rstrip('/')
            if self.isdir(p):
                if not recursive:
                    raise ValueError('Cannot delete directory, set recursive=True')
                if osp.abspath(p) == os.getcwd():
                    raise ValueError('Cannot delete current working directory')
                shutil.rmtree(p)
            else:
                os.remove(p)

    def unstrip_protocol(self, name):
        name = self._strip_protocol(name)
        return f'file://{name}'

    def _open(self, path, mode='rb', block_size=None, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir and 'w' in mode:
            self.makedirs(self._parent(path), exist_ok=True)
        return LocalFileOpener(path, mode, fs=self, **kwargs)

    def touch(self, path, truncate=True, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir:
            self.makedirs(self._parent(path), exist_ok=True)
        if self.exists(path):
            os.utime(path, None)
        else:
            open(path, 'a').close()
        if truncate:
            os.truncate(path, 0)

    def created(self, path):
        info = self.info(path=path)
        return datetime.datetime.fromtimestamp(info['created'], tz=datetime.timezone.utc)

    def modified(self, path):
        info = self.info(path=path)
        return datetime.datetime.fromtimestamp(info['mtime'], tz=datetime.timezone.utc)

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path).rstrip('/')
        if '/' in path:
            return path.rsplit('/', 1)[0]
        else:
            return cls.root_marker

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith('file://'):
            path = path[7:]
        elif path.startswith('file:'):
            path = path[5:]
        elif path.startswith('local://'):
            path = path[8:]
        elif path.startswith('local:'):
            path = path[6:]
        return make_path_posix(path).rstrip('/') or cls.root_marker

    def _isfilestore(self):
        return True

    def chmod(self, path, mode):
        path = stringify_path(path)
        return os.chmod(path, mode)