from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
@six.python_2_unicode_compatible
class ReadTarFS(FS):
    """A readable tar file."""
    _meta = {'case_insensitive': True, 'network': False, 'read_only': True, 'supports_rename': False, 'thread_safe': True, 'unicode_paths': True, 'virtual': False}
    _typemap = type_map = {tarfile.BLKTYPE: ResourceType.block_special_file, tarfile.CHRTYPE: ResourceType.character, tarfile.DIRTYPE: ResourceType.directory, tarfile.FIFOTYPE: ResourceType.fifo, tarfile.REGTYPE: ResourceType.file, tarfile.AREGTYPE: ResourceType.file, tarfile.SYMTYPE: ResourceType.symlink, tarfile.CONTTYPE: ResourceType.file, tarfile.LNKTYPE: ResourceType.symlink}

    @errors.CreateFailed.catch_all
    def __init__(self, file, encoding='utf-8'):
        super(ReadTarFS, self).__init__()
        self._file = file
        self.encoding = encoding
        if isinstance(file, (six.text_type, six.binary_type)):
            self._tar = tarfile.open(file, mode='r')
        else:
            self._tar = tarfile.open(fileobj=file, mode='r')
        self._directory_cache = None

    @property
    def _directory_entries(self):
        """Lazy directory cache."""
        if self._directory_cache is None:
            _decode = self._decode
            _directory_entries = ((_decode(info.name).strip('/'), info) for info in self._tar)

            def _list_tar():
                for name, info in _directory_entries:
                    try:
                        _name = normpath(name)
                    except IllegalBackReference:
                        pass
                    else:
                        if _name:
                            yield (_name, info)
            self._directory_cache = OrderedDict(_list_tar())
        return self._directory_cache

    def __repr__(self):
        return 'ReadTarFS({!r})'.format(self._file)

    def __str__(self):
        return "<TarFS '{}'>".format(self._file)
    if six.PY2:

        def _encode(self, s):
            return s.encode(self.encoding)

        def _decode(self, s):
            return s.decode(self.encoding)
    else:

        def _encode(self, s):
            return s

        def _decode(self, s):
            return s

    def getinfo(self, path, namespaces=None):
        _path = relpath(self.validatepath(path))
        namespaces = namespaces or ()
        raw_info = {}
        if not _path:
            raw_info['basic'] = {'name': '', 'is_dir': True}
            if 'details' in namespaces:
                raw_info['details'] = {'type': int(ResourceType.directory)}
        else:
            try:
                implicit = False
                member = self._directory_entries[_path]
            except KeyError:
                if not self.isdir(_path):
                    raise errors.ResourceNotFound(path)
                implicit = True
                member = tarfile.TarInfo(_path)
                member.type = tarfile.DIRTYPE
            raw_info['basic'] = {'name': basename(self._decode(member.name)), 'is_dir': member.isdir()}
            if 'details' in namespaces:
                raw_info['details'] = {'size': member.size, 'type': int(self.type_map[member.type])}
                if not implicit:
                    raw_info['details']['modified'] = member.mtime
            if 'access' in namespaces and (not implicit):
                raw_info['access'] = {'gid': member.gid, 'group': member.gname, 'permissions': Permissions(mode=member.mode).dump(), 'uid': member.uid, 'user': member.uname}
            if 'tar' in namespaces and (not implicit):
                raw_info['tar'] = _get_member_info(member, self.encoding)
                raw_info['tar'].update({k.replace('is', 'is_'): getattr(member, k)() for k in dir(member) if k.startswith('is')})
        return Info(raw_info)

    def isdir(self, path):
        _path = relpath(self.validatepath(path))
        try:
            return self._directory_entries[_path].isdir()
        except KeyError:
            return any((isbase(_path, name) for name in self._directory_entries))

    def isfile(self, path):
        _path = relpath(self.validatepath(path))
        try:
            return self._directory_entries[_path].isfile()
        except KeyError:
            return False

    def setinfo(self, path, info):
        self.check()
        raise errors.ResourceReadOnly(path)

    def listdir(self, path):
        _path = relpath(self.validatepath(path))
        if not self.gettype(path) is ResourceType.directory:
            raise errors.DirectoryExpected(path)
        children = (frombase(_path, n) for n in self._directory_entries if isbase(_path, n))
        content = (parts(child)[1] for child in children if relpath(child))
        return list(OrderedDict.fromkeys(content))

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        raise errors.ResourceReadOnly(path)

    def openbin(self, path, mode='r', buffering=-1, **options):
        _path = relpath(self.validatepath(path))
        if 'w' in mode or '+' in mode or 'a' in mode:
            raise errors.ResourceReadOnly(path)
        try:
            member = self._directory_entries[_path]
        except KeyError:
            six.raise_from(errors.ResourceNotFound(path), None)
        if not member.isfile():
            raise errors.FileExpected(path)
        rw = RawWrapper(cast(IO, self._tar.extractfile(member)))
        if six.PY2:

            def _flush():
                pass
            rw.flush = _flush
        return rw

    def remove(self, path):
        self.check()
        raise errors.ResourceReadOnly(path)

    def removedir(self, path):
        self.check()
        raise errors.ResourceReadOnly(path)

    def close(self):
        super(ReadTarFS, self).close()
        if hasattr(self, '_tar'):
            self._tar.close()

    def isclosed(self):
        return self._tar.closed

    def geturl(self, path, purpose='download'):
        if purpose == 'fs' and isinstance(self._file, six.string_types):
            quoted_file = url_quote(self._file)
            quoted_path = url_quote(path)
            return 'tar://{}!/{}'.format(quoted_file, quoted_path)
        else:
            raise NoURL(path, purpose)