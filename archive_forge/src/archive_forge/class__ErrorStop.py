import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class _ErrorStop(Empty):

    def __init__(self, *args, **kwargs):
        super(And._ErrorStop, self).__init__(*args, **kwargs)
        self.name = '-'
        self.leaveWhitespace()