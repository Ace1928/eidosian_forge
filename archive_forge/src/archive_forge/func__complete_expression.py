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
def _complete_expression(self, text, line, begidx, endidx):
    if not self.curframe:
        return []
    ns = {**self.curframe.f_globals, **self.curframe_locals}
    if '.' in text:
        dotted = text.split('.')
        try:
            obj = ns[dotted[0]]
            for part in dotted[1:-1]:
                obj = getattr(obj, part)
        except (KeyError, AttributeError):
            return []
        prefix = '.'.join(dotted[:-1]) + '.'
        return [prefix + n for n in dir(obj) if n.startswith(dotted[-1])]
    else:
        return [n for n in ns.keys() if n.startswith(text)]