import doctest
import importlib
import inspect
import os
import sys
import types
import unittest as pyunit
import warnings
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from typing import Callable, Generator, List, Optional, TextIO, Type, Union
from zope.interface import implementer
from attrs import define
from typing_extensions import ParamSpec, Protocol, TypeAlias, TypeGuard
from twisted.internet import defer
from twisted.python import failure, filepath, log, modules, reflect
from twisted.trial import unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator, _iterateTests
from twisted.trial._synctest import _logObserver
from twisted.trial.itrial import ITestCase
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.unittest import TestSuite
from . import itrial
def _runWithoutDecoration(self, test: Union[pyunit.TestCase, pyunit.TestSuite], forceGarbageCollection: bool=False) -> itrial.IReporter:
    """
        Private helper that runs the given test but doesn't decorate it.
        """
    result = self._makeResult()
    suite = TrialSuite([test], forceGarbageCollection)
    if self.mode == self.DRY_RUN:
        for single in _iterateTests(suite):
            result.startTest(single)
            result.addSuccess(single)
            result.stopTest(single)
    else:
        if self.mode == self.DEBUG:
            assert self.debugger is not None
            run = lambda: self.debugger.runcall(suite.run, result)
        else:
            run = lambda: suite.run(result)
        with _testDirectory(self.workingDirectory), _logFile(self.logfile):
            run()
    result.done()
    return result