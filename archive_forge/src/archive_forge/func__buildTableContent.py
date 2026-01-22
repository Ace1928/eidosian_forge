from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def _buildTableContent(self, elements):
    """
        Build a table content using C{self.linePattern} and giving elements odd
        and even classes.
        """
    tableContent = []
    rowClasses = itertools.cycle(['odd', 'even'])
    for element, rowClass in zip(elements, rowClasses):
        element['class'] = rowClass
        tableContent.append(self.linePattern % element)
    return tableContent