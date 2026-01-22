import os
import sys
import stat
import fnmatch
import collections
import errno
def _unpack_zipfile(filename, extract_dir):
    """Unpack zip `filename` to `extract_dir`
    """
    import zipfile
    if not zipfile.is_zipfile(filename):
        raise ReadError('%s is not a zip file' % filename)
    zip = zipfile.ZipFile(filename)
    try:
        for info in zip.infolist():
            name = info.filename
            if name.startswith('/') or '..' in name:
                continue
            targetpath = os.path.join(extract_dir, *name.split('/'))
            if not targetpath:
                continue
            _ensure_directory(targetpath)
            if not name.endswith('/'):
                with zip.open(name, 'r') as source, open(targetpath, 'wb') as target:
                    copyfileobj(source, target)
    finally:
        zip.close()