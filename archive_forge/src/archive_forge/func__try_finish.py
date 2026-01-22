import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
def _try_finish(self):
    assert not self._finished
    if self._returncode is None:
        return
    if all((p is not None and p.disconnected for p in self._pipes.values())):
        self._finished = True
        self._call(self._call_connection_lost, None)