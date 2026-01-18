import sys
import threading
import time
import traceback
from types import SimpleNamespace
def threading_excepthook(self, args):
    return self._excepthook(args, use_thread_hook=True)