import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_origbases():
    assert dill.copy(customIntList).__orig_bases__ == customIntList.__orig_bases__