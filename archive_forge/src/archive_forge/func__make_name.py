import threading
import sys
import tempfile
import time
from . import context
from . import process
from . import util
@staticmethod
def _make_name():
    return '%s-%s' % (process.current_process()._config['semprefix'], next(SemLock._rand))