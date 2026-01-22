import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class _EventLoggerThread(threading.Thread):
    """Thread that logs events."""

    def __init__(self, queue, ev_writer, flush_secs, flush_complete, flush_sentinel, close_sentinel):
        """Creates an _EventLoggerThread.

    Args:
      queue: A CloseableQueue from which to dequeue events. The queue will be
        closed just before the thread exits, whether due to `close_sentinel` or
        any exception raised in the writing loop.
      ev_writer: An event writer. Used to log brain events for
        the visualizer.
      flush_secs: How often, in seconds, to flush the
        pending file to disk.
      flush_complete: A threading.Event that will be set whenever a flush
        operation requested via `flush_sentinel` has been completed.
      flush_sentinel: A sentinel element in queue that tells this thread to
        flush the writer and mark the current flush operation complete.
      close_sentinel: A sentinel element in queue that tells this thread to
        terminate and close the queue.
    """
        threading.Thread.__init__(self, name='EventLoggerThread')
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        self._next_event_flush_time = 0
        self._flush_complete = flush_complete
        self._flush_sentinel = flush_sentinel
        self._close_sentinel = close_sentinel
        self.failure_exc_info = ()

    def run(self):
        try:
            while True:
                event = self._queue.get()
                if event is self._close_sentinel:
                    return
                elif event is self._flush_sentinel:
                    self._ev_writer.Flush()
                    self._flush_complete.set()
                else:
                    self._ev_writer.WriteEvent(event)
                    now = time.time()
                    if now > self._next_event_flush_time:
                        self._ev_writer.Flush()
                        self._next_event_flush_time = now + self._flush_secs
        except Exception as e:
            logging.error('EventFileWriter writer thread error: %s', e)
            self.failure_exc_info = sys.exc_info()
            raise
        finally:
            self._flush_complete.set()
            self._queue.close()