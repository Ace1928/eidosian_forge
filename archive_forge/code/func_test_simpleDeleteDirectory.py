import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_simpleDeleteDirectory(self):
    """
        L{inotify.INotify} removes a directory from the watchlist when
        it's removed from the filesystem.
        """
    calls = []

    def _callback(wp, filename, mask):

        def _():
            try:
                self.assertTrue(self.inotify._isWatched(subdir))
                subdir.remove()
            except Exception:
                d.errback()

        def _eb():
            try:
                self.assertFalse(self.inotify._isWatched(subdir))
                d.callback(None)
            except Exception:
                d.errback()
        if not calls:
            calls.append(filename)
            reactor.callLater(0, _)
        else:
            reactor.callLater(0, _eb)
    checkMask = inotify.IN_ISDIR | inotify.IN_CREATE
    self.inotify.watch(self.dirname, mask=checkMask, autoAdd=True, callbacks=[_callback])
    subdir = self.dirname.child('test')
    d = defer.Deferred()
    subdir.createDirectory()
    return d