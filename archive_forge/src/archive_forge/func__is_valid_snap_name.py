import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _is_valid_snap_name(name):
    parts = name.split('@')
    return len(parts) == 2 and _is_valid_fs_name(parts[0]) and _is_valid_name_component(parts[1])