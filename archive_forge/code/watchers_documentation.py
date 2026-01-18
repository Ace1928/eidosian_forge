import errno
import logging
import logging.config
import logging.handlers
import os
import pyinotify
import stat
import time
Eventlet friendly ThreadedNotifier

        EventletFriendlyThreadedNotifier contains additional time.sleep()
        call insude loop to allow switching to other thread when eventlet
        is used.
        It can be used with eventlet and native threads as well.
        