import threading
import sys
from paste.util import filemixin
def _writeparam(self, name, v):
    self._paramwriter(name, v)