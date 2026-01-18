import codecs
import contextlib
import locale
import logging
import math
import os
from functools import partial
from typing import TextIO, Union
import dill
def trace_setup(self, pickler):
    if not dill._dill.is_dill(pickler, child=False):
        return
    if self.isEnabledFor(logging.INFO):
        pickler._trace_depth = 1
        pickler._size_stack = []
    else:
        pickler._trace_depth = None