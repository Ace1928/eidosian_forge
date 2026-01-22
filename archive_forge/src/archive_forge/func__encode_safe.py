import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
def _encode_safe(self, u):
    """Encode possible unicode string argument to 8-bit string
        in user_encoding. Unencodable characters will be replaced
        with '?'.

        :param  u:  possible unicode string.
        :return:    encoded string if u is unicode, u itself otherwise.
        """
    return u