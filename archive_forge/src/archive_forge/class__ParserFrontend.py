import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
class _ParserFrontend(Serialize):

    def _parse(self, input, start, *args):
        if start is None:
            start = self.start
            if len(start) > 1:
                raise ValueError('Lark initialized with more than 1 possible start rule. Must specify which start rule to parse', start)
            start, = start
        return self.parser.parse(input, start, *args)