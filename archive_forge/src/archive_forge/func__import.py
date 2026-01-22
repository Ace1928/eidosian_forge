import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def _import(self, scope, name):
    InstrumentedImportReplacer.actions.append(('_import', name))
    return lazy_import.ImportReplacer._import(self, scope, name)