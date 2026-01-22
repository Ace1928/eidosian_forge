import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class InstrumentedReplacer(lazy_import.ScopeReplacer):
    """Track what actions are done"""

    @staticmethod
    def use_actions(actions):
        InstrumentedReplacer.actions = actions

    def __getattribute__(self, attr):
        InstrumentedReplacer.actions.append(('__getattribute__', attr))
        return lazy_import.ScopeReplacer.__getattribute__(self, attr)

    def __call__(self, *args, **kwargs):
        InstrumentedReplacer.actions.append(('__call__', args, kwargs))
        return lazy_import.ScopeReplacer.__call__(self, *args, **kwargs)