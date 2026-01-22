from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
class _MockGC(object):

    def get_objects(self):
        return sys.getobjects(0)

    def __getattr__(self, name):
        return getattr(gc, name)