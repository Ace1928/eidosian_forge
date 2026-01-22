from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe

        Append some bytes to the output buffer.

        @param data: C{str} to be appended to the output buffer.
        @type data: C{str}.

        @raise TypeError: If C{data} is C{unicode} instead of C{str}.
        