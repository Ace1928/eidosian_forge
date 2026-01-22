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
def _remove_pending_dir(self, tmpname):
    """Remove the pending directory

        This is called if we failed to rename into place, so that the pending
        dirs don't clutter up the lockdir.
        """
    self._trace('remove %s', tmpname)
    try:
        self.transport.delete(tmpname + self.__INFO_NAME)
        self.transport.rmdir(tmpname)
    except PathError as e:
        note(gettext('error removing pending lock: %s'), e)