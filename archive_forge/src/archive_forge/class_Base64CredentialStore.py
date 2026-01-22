import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class Base64CredentialStore(CredentialStore):
    __doc__ = 'Base64 credential store for the authentication.conf file'

    def decode_password(self, credentials):
        """See CredentialStore.decode_password."""
        import base64
        return base64.standard_b64decode(credentials['password'])