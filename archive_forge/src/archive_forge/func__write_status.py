import sys
import time
import warnings
from . import result
from .case import _SubTest
from .signals import registerResult
def _write_status(self, test, status):
    is_subtest = isinstance(test, _SubTest)
    if is_subtest or self._newline:
        if not self._newline:
            self.stream.writeln()
        if is_subtest:
            self.stream.write('  ')
        self.stream.write(self.getDescription(test))
        self.stream.write(' ... ')
    self.stream.writeln(status)
    self.stream.flush()
    self._newline = True