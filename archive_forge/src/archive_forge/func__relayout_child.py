import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _relayout_child(self, child, prop, val):
    """
        Propagate _relayout_child to parent

        Note: This method must match the name and signature of the
        corresponding method on BaseFigure
        """
    self._prop_set_child(child, prop, val)