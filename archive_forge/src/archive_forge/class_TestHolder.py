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
@implementer(ITestCase)
class TestHolder:
    """
    Placeholder for a L{TestCase} inside a reporter. As far as a L{TestResult}
    is concerned, this looks exactly like a unit test.
    """
    failureException = None

    def __init__(self, description):
        """
        @param description: A string to be displayed L{TestResult}.
        """
        self.description = description

    def __call__(self, result):
        return self.run(result)

    def id(self):
        return self.description

    def countTestCases(self):
        return 0

    def run(self, result):
        """
        This test is just a placeholder. Run the test successfully.

        @param result: The C{TestResult} to store the results in.
        @type result: L{twisted.trial.itrial.IReporter}.
        """
        result.startTest(self)
        result.addSuccess(self)
        result.stopTest(self)

    def shortDescription(self):
        return self.description