from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def _testPrelude(self, testID):
    """
        Write the name of the test to the stream, indenting it appropriately.

        If the test is the first test in a new 'branch' of the tree, also
        write all of the parents in that branch.
        """
    segments = self._getPreludeSegments(testID)
    indentLevel = 0
    for seg in segments:
        if indentLevel < len(self._lastTest):
            if seg != self._lastTest[indentLevel]:
                self._write(f'{self.indent * indentLevel}{seg}\n')
        else:
            self._write(f'{self.indent * indentLevel}{seg}\n')
        indentLevel += 1
    self._lastTest = segments