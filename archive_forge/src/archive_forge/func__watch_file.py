import errno
import logging
import logging.config
import logging.handlers
import os
import pyinotify
import stat
import time
def _watch_file(self):
    mask = pyinotify.IN_MOVED_FROM | pyinotify.IN_DELETE
    watch_manager = pyinotify.WatchManager()
    handler = _FileKeeper(watched_handler=self, watched_file=self._log_file)
    notifier = _EventletThreadedNotifier(watch_manager, default_proc_fun=handler, read_freq=FastWatchedFileHandler.READ_FREQ, timeout=FastWatchedFileHandler.TIMEOUT)
    notifier.daemon = True
    watch_manager.add_watch(self._log_dir, mask)
    notifier.start()