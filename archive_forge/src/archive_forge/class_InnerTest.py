import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class InnerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ordering.append('inner setup')
        cls.addClassCleanup(ordering.append, 'inner cleanup')

    def test(self):
        ordering.append('inner test')