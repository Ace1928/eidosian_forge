from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def tpt_callback():
    return 42