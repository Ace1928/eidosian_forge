import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_setattr(self):

    class Replaced:
        pass

    def factory(*args):
        return Replaced()
    replacer = lazy_import.ScopeReplacer({}, factory, 'name')

    def racer():
        replacer.foo = 42
    self.run_race(racer)