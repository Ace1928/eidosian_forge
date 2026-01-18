import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_simpleSubdirectoryAutoAdd(self):
    """
        L{inotify.INotify} when initialized with autoAdd==True adds
        also adds the created subdirectories to the watchlist.
        """

    def _callback(wp, filename, mask):

        def _():
            try:
                self.assertTrue(self.inotify._isWatched(subdir))
                d.callback(None)
            except Exception:
                d.errback()
        reactor.callLater(0, _)
    checkMask = inotify.IN_ISDIR | inotify.IN_CREATE
    self.inotify.watch(self.dirname, mask=checkMask, autoAdd=True, callbacks=[_callback])
    subdir = self.dirname.child('test')
    d = defer.Deferred()
    subdir.createDirectory()
    return d