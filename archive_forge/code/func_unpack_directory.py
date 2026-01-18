import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
def unpack_directory(filename, extract_dir, progress_filter=default_filter):
    """ "Unpack" a directory, using the same interface as for archives

    Raises ``UnrecognizedFormat`` if `filename` is not a directory
    """
    if not os.path.isdir(filename):
        raise UnrecognizedFormat('%s is not a directory' % filename)
    paths = {filename: ('', extract_dir)}
    for base, dirs, files in os.walk(filename):
        src, dst = paths[base]
        for d in dirs:
            paths[os.path.join(base, d)] = (src + d + '/', os.path.join(dst, d))
        for f in files:
            target = os.path.join(dst, f)
            target = progress_filter(src + f, target)
            if not target:
                continue
            ensure_directory(target)
            f = os.path.join(base, f)
            shutil.copyfile(f, target)
            shutil.copystat(f, target)