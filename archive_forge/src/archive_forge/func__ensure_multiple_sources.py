import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _ensure_multiple_sources(paths, filesystem=None):
    """
    Treat a list of paths as files belonging to a single file system

    If the file system is local then also validates that all paths
    are referencing existing *files* otherwise any non-file paths will be
    silently skipped (for example on a remote filesystem).

    Parameters
    ----------
    paths : list of path-like
        Note that URIs are not allowed.
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str)
        File system object and a list of normalized paths.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    IOError
        If the file system is local and a referenced path is not available or
        not a file.
    """
    from pyarrow.fs import LocalFileSystem, SubTreeFileSystem, _MockFileSystem, FileType, _ensure_filesystem
    if filesystem is None:
        filesystem = LocalFileSystem()
    else:
        filesystem = _ensure_filesystem(filesystem)
    is_local = isinstance(filesystem, (LocalFileSystem, _MockFileSystem)) or (isinstance(filesystem, SubTreeFileSystem) and isinstance(filesystem.base_fs, LocalFileSystem))
    paths = [filesystem.normalize_path(_stringify_path(p)) for p in paths]
    if is_local:
        for info in filesystem.get_file_info(paths):
            file_type = info.type
            if file_type == FileType.File:
                continue
            elif file_type == FileType.NotFound:
                raise FileNotFoundError(info.path)
            elif file_type == FileType.Directory:
                raise IsADirectoryError('Path {} points to a directory, but only file paths are supported. To construct a nested or union dataset pass a list of dataset objects instead.'.format(info.path))
            else:
                raise IOError('Path {} exists but its type is unknown (could be a special file such as a Unix socket or character device, or Windows NUL / CON / ...)'.format(info.path))
    return (filesystem, paths)