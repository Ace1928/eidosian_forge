import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import UnsupportedWheel
def wheel_dist_info_dir(source: ZipFile, name: str) -> str:
    """Returns the name of the contained .dist-info directory.

    Raises AssertionError or UnsupportedWheel if not found, >1 found, or
    it doesn't match the provided name.
    """
    subdirs = {p.split('/', 1)[0] for p in source.namelist()}
    info_dirs = [s for s in subdirs if s.endswith('.dist-info')]
    if not info_dirs:
        raise UnsupportedWheel('.dist-info directory not found')
    if len(info_dirs) > 1:
        raise UnsupportedWheel('multiple .dist-info directories found: {}'.format(', '.join(info_dirs)))
    info_dir = info_dirs[0]
    info_dir_name = canonicalize_name(info_dir)
    canonical_name = canonicalize_name(name)
    if not info_dir_name.startswith(canonical_name):
        raise UnsupportedWheel(f'.dist-info directory {info_dir!r} does not start with {canonical_name!r}')
    return info_dir