from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
class LeakCheckError(AssertionError):
    pass