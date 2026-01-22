import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _validate_fs_name(name):
    if not _is_valid_fs_name(name):
        raise lzc_exc.FilesystemNameInvalid(name)
    elif len(name) > MAXNAMELEN:
        raise lzc_exc.NameTooLong(name)