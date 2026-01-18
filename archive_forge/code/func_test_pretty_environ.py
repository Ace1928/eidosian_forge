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
def test_pretty_environ():
    dict_repr = pretty.pretty(dict(os.environ))
    dict_indented = dict_repr.replace('\n', '\n' + ' ' * len('environ'))
    env_repr = pretty.pretty(os.environ)
    assert env_repr == 'environ' + dict_indented