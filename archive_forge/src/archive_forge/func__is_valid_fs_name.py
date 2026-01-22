import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _is_valid_fs_name(name):
    return name and all((_is_valid_name_component(c) for c in name.split('/')))