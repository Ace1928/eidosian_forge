import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class NoMessageSupplied(errors.BzrError):
    _fmt = 'No message supplied.'