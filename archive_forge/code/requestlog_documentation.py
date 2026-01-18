import sys
from datetime import datetime
from threading import Thread
import Queue
from boto.utils import RequestHook
from boto.compat import long_type

    This class implements a request logger that uses a single thread to
    write to a log file.
    