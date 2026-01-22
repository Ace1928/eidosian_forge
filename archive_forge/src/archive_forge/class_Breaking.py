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
class Breaking(object):

    def _repr_pretty_(self, p, cycle):
        with p.group(4, 'TG: ', ':'):
            p.text('Breaking(')
            p.break_()
            p.text(')')