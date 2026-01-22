import threading
import sys
from paste.util import filemixin
def _writedefault(self, name, v):
    self._default.write(v)