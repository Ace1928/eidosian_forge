from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
class FSSpecHandler(FileSystemHandler):
    """
    Handler for fsspec-based Python filesystems.

    https://filesystem-spec.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    fs : FSSpec-compliant filesystem instance

    Examples
    --------
    >>> PyFileSystem(FSSpecHandler(fsspec_fs)) # doctest: +SKIP
    """

    def __init__(self, fs):
        self.fs = fs

    def __eq__(self, other):
        if isinstance(other, FSSpecHandler):
            return self.fs == other.fs
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FSSpecHandler):
            return self.fs != other.fs
        return NotImplemented

    def get_type_name(self):
        protocol = self.fs.protocol
        if isinstance(protocol, list):
            protocol = protocol[0]
        return 'fsspec+{0}'.format(protocol)

    def normalize_path(self, path):
        return path

    @staticmethod
    def _create_file_info(path, info):
        size = info['size']
        if info['type'] == 'file':
            ftype = FileType.File
        elif info['type'] == 'directory':
            ftype = FileType.Directory
            size = None
        else:
            ftype = FileType.Unknown
        return FileInfo(path, ftype, size=size, mtime=info.get('mtime', None))

    def get_file_info(self, paths):
        infos = []
        for path in paths:
            try:
                info = self.fs.info(path)
            except FileNotFoundError:
                infos.append(FileInfo(path, FileType.NotFound))
            else:
                infos.append(self._create_file_info(path, info))
        return infos

    def get_file_info_selector(self, selector):
        if not self.fs.isdir(selector.base_dir):
            if self.fs.exists(selector.base_dir):
                raise NotADirectoryError(selector.base_dir)
            elif selector.allow_not_found:
                return []
            else:
                raise FileNotFoundError(selector.base_dir)
        if selector.recursive:
            maxdepth = None
        else:
            maxdepth = 1
        infos = []
        selected_files = self.fs.find(selector.base_dir, maxdepth=maxdepth, withdirs=True, detail=True)
        for path, info in selected_files.items():
            _path = path.strip('/')
            base_dir = selector.base_dir.strip('/')
            if _path != base_dir:
                infos.append(self._create_file_info(path, info))
        return infos

    def create_dir(self, path, recursive):
        try:
            self.fs.mkdir(path, create_parents=recursive)
        except FileExistsError:
            pass

    def delete_dir(self, path):
        self.fs.rm(path, recursive=True)

    def _delete_dir_contents(self, path, missing_dir_ok):
        try:
            subpaths = self.fs.listdir(path, detail=False)
        except FileNotFoundError:
            if missing_dir_ok:
                return
            raise
        for subpath in subpaths:
            if self.fs.isdir(subpath):
                self.fs.rm(subpath, recursive=True)
            elif self.fs.isfile(subpath):
                self.fs.rm(subpath)

    def delete_dir_contents(self, path, missing_dir_ok):
        if path.strip('/') == '':
            raise ValueError("delete_dir_contents called on path '", path, "'")
        self._delete_dir_contents(path, missing_dir_ok)

    def delete_root_dir_contents(self):
        self._delete_dir_contents('/')

    def delete_file(self, path):
        if not self.fs.exists(path):
            raise FileNotFoundError(path)
        self.fs.rm(path)

    def move(self, src, dest):
        self.fs.mv(src, dest, recursive=True)

    def copy_file(self, src, dest):
        self.fs.copy(src, dest)

    def open_input_stream(self, path):
        from pyarrow import PythonFile
        if not self.fs.isfile(path):
            raise FileNotFoundError(path)
        return PythonFile(self.fs.open(path, mode='rb'), mode='r')

    def open_input_file(self, path):
        from pyarrow import PythonFile
        if not self.fs.isfile(path):
            raise FileNotFoundError(path)
        return PythonFile(self.fs.open(path, mode='rb'), mode='r')

    def open_output_stream(self, path, metadata):
        from pyarrow import PythonFile
        return PythonFile(self.fs.open(path, mode='wb'), mode='w')

    def open_append_stream(self, path, metadata):
        from pyarrow import PythonFile
        return PythonFile(self.fs.open(path, mode='ab'), mode='w')