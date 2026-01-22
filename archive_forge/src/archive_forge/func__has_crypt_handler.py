import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def _has_crypt_handler(name, loaded_only=False):
    """check if handler name is known.

    this is only useful for two cases:

    * quickly checking if handler has already been loaded
    * checking if handler exists, without actually loading it

    :arg name: name of handler
    :param loaded_only: if ``True``, returns False if handler exists but hasn't been loaded
    """
    return name in _handlers or (not loaded_only and name in _locations)