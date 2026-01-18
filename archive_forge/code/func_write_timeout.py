from __future__ import absolute_import
import io
import time
@write_timeout.setter
def write_timeout(self, timeout):
    """Change timeout setting."""
    if timeout is not None:
        if timeout < 0:
            raise ValueError('Not a valid timeout: {!r}'.format(timeout))
        try:
            timeout + 1
        except TypeError:
            raise ValueError('Not a valid timeout: {!r}'.format(timeout))
    self._write_timeout = timeout
    if self.is_open:
        self._reconfigure_port()