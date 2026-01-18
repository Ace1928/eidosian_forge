import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def write_c14n(self, file):
    return self.write(file, method='c14n')