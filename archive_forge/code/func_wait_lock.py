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
def wait_lock(self, timeout=None, poll=None, max_attempts=None):
    """Wait a certain period for a lock.

        If the lock can be acquired within the bounded time, it
        is taken and this returns.  Otherwise, LockContention
        is raised.  Either way, this function should return within
        approximately `timeout` seconds.  (It may be a bit more if
        a transport operation takes a long time to complete.)

        :param timeout: Approximate maximum amount of time to wait for the
        lock, in seconds.

        :param poll: Delay in seconds between retrying the lock.

        :param max_attempts: Maximum number of times to try to lock.

        :return: The lock token.
        """
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT_SECONDS
    if poll is None:
        poll = _DEFAULT_POLL_SECONDS
    deadline = time.time() + timeout
    deadline_str = None
    last_info = None
    attempt_count = 0
    lock_url = self.lock_url_for_display()
    while True:
        attempt_count += 1
        try:
            return self.attempt_lock()
        except LockContention:
            pass
        new_info = self.peek()
        if new_info is not None and new_info != last_info:
            if last_info is None:
                start = gettext('Unable to obtain')
            else:
                start = gettext('Lock owner changed for')
            last_info = new_info
            msg = gettext('{0} lock {1} {2}.').format(start, lock_url, new_info)
            if deadline_str is None:
                deadline_str = time.strftime('%H:%M:%S', time.localtime(deadline))
            if timeout > 0:
                msg += '\n' + gettext('Will continue to try until %s, unless you press Ctrl-C.') % deadline_str
            msg += '\n' + gettext('See "brz help break-lock" for more.')
            self._report_function(msg)
        if max_attempts is not None and attempt_count >= max_attempts:
            self._trace('exceeded %d attempts')
            raise LockContention(self)
        if time.time() + poll < deadline:
            self._trace('waiting %ss', poll)
            time.sleep(poll)
        else:
            self._trace('timeout after waiting %ss', timeout)
            raise LockContention('(local)', lock_url)