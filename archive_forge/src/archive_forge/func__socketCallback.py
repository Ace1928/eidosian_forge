from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _socketCallback(self, cfSocket, callbackType, ignoredAddress, ignoredData, context):
    """
        The socket callback issued by CFRunLoop.  This will issue C{doRead} or
        C{doWrite} calls to the L{IReadDescriptor} and L{IWriteDescriptor}
        registered with the file descriptor that we are being notified of.

        @param cfSocket: The C{CFSocket} which has got some activity.

        @param callbackType: The type of activity that we are being notified
            of.  Either C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}.

        @param ignoredAddress: Unused, because this is not used for either of
            the callback types we register for.

        @param ignoredData: Unused, because this is not used for either of the
            callback types we register for.

        @param context: The data associated with this callback by
            C{CFSocketCreateWithNative} (in C{CFReactor._watchFD}).  A 2-tuple
            of C{(int, CFRunLoopSource)}.
        """
    fd, smugglesrc = context
    if fd not in self._fdmap:
        CFRunLoopRemoveSource(self._cfrunloop, smugglesrc, kCFRunLoopCommonModes)
        return
    src, skt, readWriteDescriptor, rw = self._fdmap[fd]

    def _drdw():
        why = None
        isRead = False
        try:
            if readWriteDescriptor.fileno() == -1:
                why = _NO_FILEDESC
            else:
                isRead = callbackType == kCFSocketReadCallBack
                if isRead:
                    if rw[_READ]:
                        why = readWriteDescriptor.doRead()
                elif rw[_WRITE]:
                    why = readWriteDescriptor.doWrite()
        except BaseException:
            why = sys.exc_info()[1]
            log.err()
        if why:
            self._disconnectSelectable(readWriteDescriptor, why, isRead)
    log.callWithLogger(readWriteDescriptor, _drdw)