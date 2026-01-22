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
def _getPreludeSegments(self, testID):
    """
        Return a list of all non-leaf segments to display in the tree.

        Normally this is the module and class name.
        """
    segments = testID.split('.')[:-1]
    if len(segments) == 0:
        return segments
    segments = [seg for seg in ('.'.join(segments[:-1]), segments[-1]) if len(seg) > 0]
    return segments