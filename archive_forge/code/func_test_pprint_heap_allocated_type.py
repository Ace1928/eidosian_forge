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
def test_pprint_heap_allocated_type():
    """
    Test that pprint works for heap allocated types.
    """
    module_name = 'xxlimited_35'
    expected_output = 'xxlimited.Null' if sys.version_info < (3, 10, 6) else 'xxlimited_35.Null'
    xxlimited = pytest.importorskip(module_name)
    output = pretty.pretty(xxlimited.Null)
    assert output == expected_output