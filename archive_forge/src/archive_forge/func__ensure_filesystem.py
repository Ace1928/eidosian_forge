from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
def _ensure_filesystem(filesystem, use_mmap=False, allow_legacy_filesystem=False):
    if isinstance(filesystem, FileSystem):
        return filesystem
    elif isinstance(filesystem, str):
        if use_mmap:
            raise ValueError('Specifying to use memory mapping not supported for filesystem specified as an URI string')
        return _filesystem_from_str(filesystem)
    try:
        import fsspec
    except ImportError:
        pass
    else:
        if isinstance(filesystem, fsspec.AbstractFileSystem):
            if type(filesystem).__name__ == 'LocalFileSystem':
                return LocalFileSystem(use_mmap=use_mmap)
            return PyFileSystem(FSSpecHandler(filesystem))
    import pyarrow.filesystem as legacyfs
    if isinstance(filesystem, legacyfs.LocalFileSystem):
        return LocalFileSystem(use_mmap=use_mmap)
    if allow_legacy_filesystem and isinstance(filesystem, legacyfs.FileSystem):
        return filesystem
    raise TypeError("Unrecognized filesystem: {}. `filesystem` argument must be a FileSystem instance or a valid file system URI'".format(type(filesystem)))