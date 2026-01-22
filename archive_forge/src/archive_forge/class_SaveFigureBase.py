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
class SaveFigureBase(ToolBase):
    """Base tool for figure saving."""
    description = 'Save the figure'
    image = 'filesave'
    default_keymap = property(lambda self: mpl.rcParams['keymap.save'])