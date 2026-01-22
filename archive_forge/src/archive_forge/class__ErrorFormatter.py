import os
import re
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
class _ErrorFormatter:
    """
    Formatter for Windows error messages.

    @ivar winError: A callable which takes one integer error number argument
        and returns a L{WindowsError} instance for that error (like
        L{ctypes.WinError}).

    @ivar formatMessage: A callable which takes one integer error number
        argument and returns a C{str} giving the message for that error (like
        U{win32api.FormatMessage<http://
        timgolden.me.uk/pywin32-docs/win32api__FormatMessage_meth.html>}).

    @ivar errorTab: A mapping from integer error numbers to C{str} messages
        which correspond to those erorrs (like I{socket.errorTab}).
    """

    def __init__(self, WinError, FormatMessage, errorTab):
        self.winError = WinError
        self.formatMessage = FormatMessage
        self.errorTab = errorTab

    @classmethod
    def fromEnvironment(cls):
        """
        Get as many of the platform-specific error translation objects as
        possible and return an instance of C{cls} created with them.
        """
        try:
            from ctypes import WinError
        except ImportError:
            WinError = None
        try:
            from win32api import FormatMessage
        except ImportError:
            FormatMessage = None
        try:
            from socket import errorTab
        except ImportError:
            errorTab = None
        return cls(WinError, FormatMessage, errorTab)

    def formatError(self, errorcode):
        """
        Returns the string associated with a Windows error message, such as the
        ones found in socket.error.

        Attempts direct lookup against the win32 API via ctypes and then
        pywin32 if available), then in the error table in the socket module,
        then finally defaulting to C{os.strerror}.

        @param errorcode: the Windows error code
        @type errorcode: C{int}

        @return: The error message string
        @rtype: C{str}
        """
        if self.winError is not None:
            return self.winError(errorcode).strerror
        if self.formatMessage is not None:
            return self.formatMessage(errorcode)
        if self.errorTab is not None:
            result = self.errorTab.get(errorcode)
            if result is not None:
                return result
        return os.strerror(errorcode)