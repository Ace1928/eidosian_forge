import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
def _process_exited(self, returncode):
    assert returncode is not None, returncode
    assert self._returncode is None, self._returncode
    if self._loop.get_debug():
        logger.info('%r exited with return code %r', self, returncode)
    self._returncode = returncode
    if self._proc.returncode is None:
        self._proc.returncode = returncode
    self._call(self._protocol.process_exited)
    self._try_finish()