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
def test_really_bad_repr():
    with pytest.raises(BadException):
        pretty.pretty(ReallyBadRepr())