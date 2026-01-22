import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
def _resolve_tar_file_or_dir(tar_obj, tar_member_obj):
    """Resolve any links and extract link targets as normal files."""
    while tar_member_obj is not None and (tar_member_obj.islnk() or tar_member_obj.issym()):
        linkpath = tar_member_obj.linkname
        if tar_member_obj.issym():
            base = posixpath.dirname(tar_member_obj.name)
            linkpath = posixpath.join(base, linkpath)
            linkpath = posixpath.normpath(linkpath)
        tar_member_obj = tar_obj._getmember(linkpath)
    is_file_or_dir = tar_member_obj is not None and (tar_member_obj.isfile() or tar_member_obj.isdir())
    if is_file_or_dir:
        return tar_member_obj
    raise LookupError('Got unknown file type')