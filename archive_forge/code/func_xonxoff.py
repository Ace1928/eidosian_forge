from __future__ import absolute_import
import io
import time
@xonxoff.setter
def xonxoff(self, xonxoff):
    """Change XON/XOFF setting."""
    self._xonxoff = xonxoff
    if self.is_open:
        self._reconfigure_port()