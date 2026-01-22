import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_indent(self, doc, indent=4):
    out = []
    for line in doc:
        out += [' ' * indent + line]
    return out