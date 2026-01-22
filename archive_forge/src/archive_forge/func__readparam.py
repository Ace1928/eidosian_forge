import threading
import sys
from paste.util import filemixin
def _readparam(self, name, size):
    self._paramreader(name, size)