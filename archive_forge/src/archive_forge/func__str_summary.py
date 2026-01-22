import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_summary(self):
    if self['Summary']:
        return self['Summary'] + ['']
    else:
        return []