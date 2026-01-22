import errno
import logging
import logging.config
import logging.handlers
import os
import pyinotify
import stat
import time
class _EventletThreadedNotifier(pyinotify.ThreadedNotifier):

    def loop(self):
        """Eventlet friendly ThreadedNotifier

        EventletFriendlyThreadedNotifier contains additional time.sleep()
        call insude loop to allow switching to other thread when eventlet
        is used.
        It can be used with eventlet and native threads as well.
        """
        while not self._stop_event.is_set():
            self.process_events()
            time.sleep(0)
            ref_time = time.time()
            if self.check_events():
                self._sleep(ref_time)
                self.read_events()