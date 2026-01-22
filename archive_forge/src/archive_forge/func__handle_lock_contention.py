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
def _handle_lock_contention(self, other_holder):
    """A lock we want to take is held by someone else.

        This function can: tell the user about it; possibly detect that it's
        safe or appropriate to steal the lock, or just raise an exception.

        If this function returns (without raising an exception) the lock will
        be attempted again.

        :param other_holder: A LockHeldInfo for the current holder; note that
            it might be None if the lock can be seen to be held but the info
            can't be read.
        """
    if other_holder is not None:
        if other_holder.is_lock_holder_known_dead():
            if self.get_config().get('locks.steal_dead'):
                ui.ui_factory.show_user_warning('locks_steal_dead', lock_url=urlutils.join(self.transport.base, self.path), other_holder_info=str(other_holder))
                self.force_break(other_holder)
                self._trace('stole lock from dead holder')
                return
    raise LockContention(self)