import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def _read_info_file(self, path):
    """Read one given info file.

        peek() reads the info file of the lock holder, if any.
        """
    return LockHeldInfo.from_info_file_bytes(self.transport.get_bytes(path))