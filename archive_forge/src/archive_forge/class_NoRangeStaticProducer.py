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
class NoRangeStaticProducer(StaticProducer):
    """
    A L{StaticProducer} that writes the entire file to the request.
    """

    def start(self):
        self.request.registerProducer(self, False)

    def resumeProducing(self):
        if not self.request:
            return
        data = self.fileObject.read(self.bufferSize)
        if data:
            self.request.write(data)
        else:
            self.request.unregisterProducer()
            self.request.finish()
            self.stopProducing()