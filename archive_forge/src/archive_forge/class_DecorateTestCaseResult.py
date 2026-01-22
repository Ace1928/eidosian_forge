import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
class DecorateTestCaseResult:
    """Decorate a TestCase and permit customisation of the result for runs."""

    def __init__(self, case, callout, before_run=None, after_run=None):
        """Construct a DecorateTestCaseResult.

        :param case: The case to decorate.
        :param callout: A callback to call when run/__call__/debug is called.
            Must take a result parameter and return a result object to be used.
            For instance: lambda result: result.
        :param before_run: If set, call this with the decorated result before
            calling into the decorated run/__call__ method.
        :param before_run: If set, call this with the decorated result after
            calling into the decorated run/__call__ method.
        """
        self.decorated = case
        self.callout = callout
        self.before_run = before_run
        self.after_run = after_run

    def _run(self, result, run_method):
        result = self.callout(result)
        if self.before_run:
            self.before_run(result)
        try:
            return run_method(result)
        finally:
            if self.after_run:
                self.after_run(result)

    def run(self, result=None):
        self._run(result, self.decorated.run)

    def __call__(self, result=None):
        self._run(result, self.decorated)

    def __getattr__(self, name):
        return getattr(self.decorated, name)

    def __delattr__(self, name):
        delattr(self.decorated, name)

    def __setattr__(self, name, value):
        if name in ('decorated', 'callout', 'before_run', 'after_run'):
            self.__dict__[name] = value
            return
        setattr(self.decorated, name, value)