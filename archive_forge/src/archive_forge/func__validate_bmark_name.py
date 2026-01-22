import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _validate_bmark_name(name):
    if not _is_valid_bmark_name(name):
        raise lzc_exc.BookmarkNameInvalid(name)
    elif len(name) > MAXNAMELEN:
        raise lzc_exc.NameTooLong(name)