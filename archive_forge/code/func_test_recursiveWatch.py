import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_recursiveWatch(self):
    """
        L{inotify.INotify.watch} with recursive==True will add all the
        subdirectories under the given path to the watchlist.
        """
    subdir = self.dirname.child('test')
    subdir2 = subdir.child('test2')
    subdir3 = subdir2.child('test3')
    subdir3.makedirs()
    dirs = [subdir, subdir2, subdir3]
    self.inotify.watch(self.dirname, recursive=True)
    self.inotify.watch(self.dirname, recursive=True)
    for d in dirs:
        self.assertTrue(self.inotify._isWatched(d))