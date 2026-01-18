import sys
import time
import warnings
from . import result
from .case import _SubTest
from .signals import registerResult
def writeln(self, arg=None):
    if arg:
        self.write(arg)
    self.write('\n')