import functools
from cupy import cuda
from cupy_backends.cuda.api import runtime
def _recreate_cm(self, message):
    if self.message is None:
        self.message = message
    return self