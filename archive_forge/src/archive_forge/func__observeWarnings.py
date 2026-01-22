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
def _observeWarnings(self, event):
    """
        Observe warning events and write them to C{self._stream}.

        This method is a log observer which will be registered with
        C{self._publisher.addObserver}.

        @param event: A C{dict} from the logging system.  If it has a
            C{'warning'} key, a logged warning will be extracted from it and
            possibly written to C{self.stream}.
        """
    if 'warning' in event:
        key = (event['filename'], event['lineno'], event['category'].split('.')[-1], str(event['warning']))
        if key not in self._warningCache:
            self._warningCache.add(key)
            self._stream.write('%s:%s: %s: %s\n' % key)