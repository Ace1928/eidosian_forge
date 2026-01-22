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
class LoggedSuite(TestSuite):
    """
    Any errors logged in this suite will be reported to the L{TestResult}
    object.
    """

    def run(self, result):
        """
        Run the suite, storing all errors in C{result}. If an error is logged
        while no tests are running, then it will be added as an error to
        C{result}.

        @param result: A L{TestResult} object.
        """
        observer = _logObserver
        observer._add()
        super().run(result)
        observer._remove()
        for error in observer.getErrors():
            result.addError(TestHolder(NOT_IN_TEST), error)
        observer.flushErrors()