from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
class BreakingRepr(object):

    def __repr__(self):
        return 'Breaking(\n)'