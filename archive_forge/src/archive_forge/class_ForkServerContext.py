import os
import sys
import threading
from . import process
from . import reduction
class ForkServerContext(BaseContext):
    _name = 'forkserver'
    Process = ForkServerProcess

    def _check_available(self):
        if not reduction.HAVE_SEND_HANDLE:
            raise ValueError('forkserver start method not available')