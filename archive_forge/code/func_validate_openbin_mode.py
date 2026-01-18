from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def validate_openbin_mode(mode, _valid_chars=frozenset('rwxab+')):
    """Check ``mode`` parameter of `~fs.base.FS.openbin` is valid.

    Arguments:
        mode (str): Mode parameter.

    Raises:
        `ValueError` if mode is not valid.

    """
    if 't' in mode:
        raise ValueError('text mode not valid in openbin')
    if not mode:
        raise ValueError('mode must not be empty')
    if mode[0] not in 'rwxa':
        raise ValueError("mode must start with 'r', 'w', 'a' or 'x'")
    if not _valid_chars.issuperset(mode):
        raise ValueError("mode '{}' contains invalid characters".format(mode))