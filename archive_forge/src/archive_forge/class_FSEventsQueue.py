import os
import logging
import unicodedata
from threading import Thread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
import AppKit
from FSEvents import (
from FSEvents import (
class FSEventsQueue(Thread):
    """ Low level FSEvents client. """

    def __init__(self, path):
        Thread.__init__(self)
        self._queue = queue.Queue()
        self._run_loop = None
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        self._path = unicodedata.normalize('NFC', path)
        context = None
        latency = 1.0
        self._stream_ref = FSEventStreamCreate(kCFAllocatorDefault, self._callback, context, [self._path], kFSEventStreamEventIdSinceNow, latency, kFSEventStreamCreateFlagNoDefer | kFSEventStreamCreateFlagFileEvents)
        if self._stream_ref is None:
            raise IOError('FSEvents. Could not create stream.')

    def run(self):
        pool = AppKit.NSAutoreleasePool.alloc().init()
        self._run_loop = CFRunLoopGetCurrent()
        FSEventStreamScheduleWithRunLoop(self._stream_ref, self._run_loop, kCFRunLoopDefaultMode)
        if not FSEventStreamStart(self._stream_ref):
            FSEventStreamInvalidate(self._stream_ref)
            FSEventStreamRelease(self._stream_ref)
            raise IOError('FSEvents. Could not start stream.')
        CFRunLoopRun()
        FSEventStreamStop(self._stream_ref)
        FSEventStreamInvalidate(self._stream_ref)
        FSEventStreamRelease(self._stream_ref)
        del pool
        self._queue.put(None)

    def stop(self):
        if self._run_loop is not None:
            CFRunLoopStop(self._run_loop)

    def _callback(self, streamRef, clientCallBackInfo, numEvents, eventPaths, eventFlags, eventIDs):
        events = [NativeEvent(path, flags, _id) for path, flags, _id in zip(eventPaths, eventFlags, eventIDs)]
        logger.debug('FSEvents callback. Got %d events:' % numEvents)
        for e in events:
            logger.debug(e)
        self._queue.put(events)

    def read_events(self):
        """
        Returns a list or one or more events, or None if there are no more
        events to be read.
        """
        if not self.is_alive():
            return None
        return self._queue.get()