import errno
import logging
import logging.config
import logging.handlers
import os
import pyinotify
import stat
import time
class FastWatchedFileHandler(logging.handlers.WatchedFileHandler, object):
    """Frequency of reading events.

    Watching thread sleeps max(0, READ_FREQ - (TIMEOUT / 1000)) seconds.
    """
    READ_FREQ = 5
    'Poll timeout in milliseconds.\n\n    See https://docs.python.org/2/library/select.html#select.poll.poll'
    TIMEOUT = 5

    def __init__(self, logpath, *args, **kwargs):
        self._log_file = os.path.basename(logpath)
        self._log_dir = os.path.dirname(logpath)
        super(FastWatchedFileHandler, self).__init__(logpath, *args, **kwargs)
        self._watch_file()

    def _watch_file(self):
        mask = pyinotify.IN_MOVED_FROM | pyinotify.IN_DELETE
        watch_manager = pyinotify.WatchManager()
        handler = _FileKeeper(watched_handler=self, watched_file=self._log_file)
        notifier = _EventletThreadedNotifier(watch_manager, default_proc_fun=handler, read_freq=FastWatchedFileHandler.READ_FREQ, timeout=FastWatchedFileHandler.TIMEOUT)
        notifier.daemon = True
        watch_manager.add_watch(self._log_dir, mask)
        notifier.start()

    def reopen_file(self):
        try:
            sres = os.stat(self.baseFilename)
        except OSError as err:
            if err.errno == errno.ENOENT:
                sres = None
            else:
                raise
        if not sres or sres[stat.ST_DEV] != self.dev or sres[stat.ST_INO] != self.ino:
            if self.stream is not None:
                self.stream.flush()
                self.stream.close()
                self.stream = None
                self.stream = self._open()
                self._statstream()