from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
class ignores_types_in_strict_leakcheck(object):

    def __init__(self, types):
        self.types = types

    def __call__(self, func):
        func.leakcheck_ignore_types = self.types
        return func