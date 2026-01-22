from __future__ import absolute_import
import unittest
import sys
import os
class TestMissingSpeedups(unittest.TestCase):

    def runTest(self):
        if hasattr(sys, 'pypy_translation_info'):
            "PyPy doesn't need speedups! :)"
        elif hasattr(self, 'skipTest'):
            self.skipTest('_speedups.so is missing!')