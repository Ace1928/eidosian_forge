import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _notificationTest(self, mask, operation, expectedPath=None):
    """
        Test notification from some filesystem operation.

        @param mask: The event mask to use when setting up the watch.

        @param operation: A function which will be called with the
            name of a file in the watched directory and which should
            trigger the event.

        @param expectedPath: Optionally, the name of the path which is
            expected to come back in the notification event; this will
            also be passed to C{operation} (primarily useful when the
            operation is being done to the directory itself, not a
            file in it).

        @return: A L{Deferred} which fires successfully when the
            expected event has been received or fails otherwise.
        """
    if expectedPath is None:
        expectedPath = self.dirname.child('foo.bar')
    notified = defer.Deferred()

    def cbNotified(result):
        watch, filename, events = result
        self.assertEqual(filename.asBytesMode(), expectedPath.asBytesMode())
        self.assertTrue(events & mask)
    notified.addCallback(cbNotified)
    self.inotify.watch(self.dirname, mask=mask, callbacks=[lambda *args: notified.callback(args)])
    operation(expectedPath)
    return notified