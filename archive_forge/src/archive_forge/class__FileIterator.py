import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
class _FileIterator:
    """Return an iterator which crawls over a stream of lines with a function (PRIVATE).

    The generator function is expected to yield a tuple, while
    consuming input
    """

    def __init__(self, func, fname, handle=None):
        self.func = func
        if handle is None:
            self.stream = open(fname)
        else:
            self.stream = handle
        self.fname = fname
        self.done = False

    def __iter__(self):
        if self.done:
            self.done = True
            raise StopIteration
        return self

    def __next__(self):
        return self.func(self)

    def __del__(self):
        self.stream.close()
        os.remove(self.fname)