import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo
from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
from pip._internal.utils.misc import ensure_dir
def untar_file(filename: str, location: str) -> None:
    """
    Untar the file (with path `filename`) to the destination `location`.
    All files are written based on system defaults and umask (i.e. permissions
    are not preserved), except that regular file members with any execute
    permissions (user, group, or world) have "chmod +x" applied after being
    written.  Note that for windows, any execute changes using os.chmod are
    no-ops per the python docs.
    """
    ensure_dir(location)
    if filename.lower().endswith('.gz') or filename.lower().endswith('.tgz'):
        mode = 'r:gz'
    elif filename.lower().endswith(BZ2_EXTENSIONS):
        mode = 'r:bz2'
    elif filename.lower().endswith(XZ_EXTENSIONS):
        mode = 'r:xz'
    elif filename.lower().endswith('.tar'):
        mode = 'r'
    else:
        logger.warning('Cannot determine compression type for file %s', filename)
        mode = 'r:*'
    tar = tarfile.open(filename, mode, encoding='utf-8')
    try:
        leading = has_leading_dir([member.name for member in tar.getmembers()])
        for member in tar.getmembers():
            fn = member.name
            if leading:
                fn = split_leading_dir(fn)[1]
            path = os.path.join(location, fn)
            if not is_within_directory(location, path):
                message = 'The tar file ({}) has a file ({}) trying to install outside target directory ({})'
                raise InstallationError(message.format(filename, path, location))
            if member.isdir():
                ensure_dir(path)
            elif member.issym():
                try:
                    tar._extract_member(member, path)
                except Exception as exc:
                    logger.warning('In the tar file %s the member %s is invalid: %s', filename, member.name, exc)
                    continue
            else:
                try:
                    fp = tar.extractfile(member)
                except (KeyError, AttributeError) as exc:
                    logger.warning('In the tar file %s the member %s is invalid: %s', filename, member.name, exc)
                    continue
                ensure_dir(os.path.dirname(path))
                assert fp is not None
                with open(path, 'wb') as destfp:
                    shutil.copyfileobj(fp, destfp)
                fp.close()
                tar.utime(member, path)
                if member.mode & 73:
                    set_extracted_file_to_default_mode_plus_executable(path)
    finally:
        tar.close()