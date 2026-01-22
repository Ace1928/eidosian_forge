import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
def _get_client_commands(self):
    """Provide a list of commands that may invoke the mail client"""
    if sys.platform == 'win32':
        import win32utils
        return [win32utils.get_app_path(i) for i in self._client_commands]
    else:
        return self._client_commands