import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class InstrumentedImportReplacer(lazy_import.ImportReplacer):

    @staticmethod
    def use_actions(actions):
        InstrumentedImportReplacer.actions = actions

    def _import(self, scope, name):
        InstrumentedImportReplacer.actions.append(('_import', name))
        return lazy_import.ImportReplacer._import(self, scope, name)

    def __getattribute__(self, attr):
        InstrumentedImportReplacer.actions.append(('__getattribute__', attr))
        return lazy_import.ScopeReplacer.__getattribute__(self, attr)

    def __call__(self, *args, **kwargs):
        InstrumentedImportReplacer.actions.append(('__call__', args, kwargs))
        return lazy_import.ScopeReplacer.__call__(self, *args, **kwargs)