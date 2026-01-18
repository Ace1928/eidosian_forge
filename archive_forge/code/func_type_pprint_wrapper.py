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
def type_pprint_wrapper(obj, p, cycle):
    if obj is MyObj:
        type_pprint_wrapper.called = True
    return pretty._type_pprint(obj, p, cycle)