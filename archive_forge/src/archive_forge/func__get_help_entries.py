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
def _get_help_entries(self):
    return [(name, self._format_tool_keymap(name), tool.description) for name, tool in sorted(self.toolmanager.tools.items()) if tool.description]