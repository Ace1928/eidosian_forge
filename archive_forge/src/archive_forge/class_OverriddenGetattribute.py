import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
class OverriddenGetattribute:

    def __getattribute__(self, attr):
        raise AssertionError('custom __getattribute__ executed')
    a = 1