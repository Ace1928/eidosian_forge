import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
class _Watch:
    """
    Watch object that represents a Watch point in the filesystem. The
    user should let INotify to create these objects

    @ivar path: The path over which this watch point is monitoring
    @ivar mask: The events monitored by this watchpoint
    @ivar autoAdd: Flag that determines whether this watch point
        should automatically add created subdirectories
    @ivar callbacks: L{list} of callback functions that will be called
        when an event occurs on this watch.
    """

    def __init__(self, path, mask=IN_WATCH_MASK, autoAdd=False, callbacks=None):
        self.path = path.asBytesMode()
        self.mask = mask
        self.autoAdd = autoAdd
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def _notify(self, filepath, events):
        """
        Callback function used by L{INotify} to dispatch an event.
        """
        filepath = filepath.asBytesMode()
        for callback in self.callbacks:
            callback(self, filepath, events)