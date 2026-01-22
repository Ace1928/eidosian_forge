import os
import signal
from . import util
def _send_signal(self, sig):
    if self.returncode is None:
        try:
            os.kill(self.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            if self.wait(timeout=0.1) is None:
                raise