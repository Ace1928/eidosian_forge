import os
import sys
import stat
import fnmatch
import collections
import errno
def unpack_archive(filename, extract_dir=None, format=None, *, filter=None):
    """Unpack an archive.

    `filename` is the name of the archive.

    `extract_dir` is the name of the target directory, where the archive
    is unpacked. If not provided, the current working directory is used.

    `format` is the archive format: one of "zip", "tar", "gztar", "bztar",
    or "xztar".  Or any other registered format.  If not provided,
    unpack_archive will use the filename extension and see if an unpacker
    was registered for that extension.

    In case none is found, a ValueError is raised.

    If `filter` is given, it is passed to the underlying
    extraction function.
    """
    sys.audit('shutil.unpack_archive', filename, extract_dir, format)
    if extract_dir is None:
        extract_dir = os.getcwd()
    extract_dir = os.fspath(extract_dir)
    filename = os.fspath(filename)
    if filter is None:
        filter_kwargs = {}
    else:
        filter_kwargs = {'filter': filter}
    if format is not None:
        try:
            format_info = _UNPACK_FORMATS[format]
        except KeyError:
            raise ValueError("Unknown unpack format '{0}'".format(format)) from None
        func = format_info[1]
        func(filename, extract_dir, **dict(format_info[2]), **filter_kwargs)
    else:
        format = _find_unpack_format(filename)
        if format is None:
            raise ReadError("Unknown archive format '{0}'".format(filename))
        func = _UNPACK_FORMATS[format][1]
        kwargs = dict(_UNPACK_FORMATS[format][2]) | filter_kwargs
        func(filename, extract_dir, **kwargs)