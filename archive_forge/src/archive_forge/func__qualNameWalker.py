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
def _qualNameWalker(qualName):
    """
    Given a Python qualified name, this function yields a 2-tuple of the most
    specific qualified name first, followed by the next-most-specific qualified
    name, and so on, paired with the remainder of the qualified name.

    @param qualName: A Python qualified name.
    @type qualName: L{str}
    """
    yield (qualName, [])
    qualParts = qualName.split('.')
    for index in range(1, len(qualParts)):
        yield ('.'.join(qualParts[:-index]), qualParts[-index:])