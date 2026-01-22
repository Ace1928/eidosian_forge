import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
def _get_help_text(self):
    entries = self._get_help_entries()
    entries = ['{}: {}\n\t{}'.format(*entry) for entry in entries]
    return '\n'.join(entries)