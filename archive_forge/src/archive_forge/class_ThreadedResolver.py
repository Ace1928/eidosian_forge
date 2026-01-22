import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
class ThreadedResolver(_ThreadedResolverImpl):

    def __init__(self, reactor=None):
        if reactor is None:
            from twisted.internet import reactor
        _ThreadedResolverImpl.__init__(self, reactor)
        warnings.warn('twisted.names.client.ThreadedResolver is deprecated since Twisted 9.0, use twisted.internet.base.ThreadedResolver instead.', category=DeprecationWarning, stacklevel=2)