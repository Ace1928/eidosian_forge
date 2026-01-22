import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _select_frame(self, number):
    assert 0 <= number < len(self.stack)
    self.curindex = number
    self.curframe = self.stack[self.curindex][0]
    self.curframe_locals = self.curframe.f_locals
    self.print_stack_entry(self.stack[self.curindex])
    self.lineno = None