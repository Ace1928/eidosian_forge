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
def _get_prop_validator(self, prop):
    """
        Custom _get_prop_validator that handles subplot properties
        """
    prop = self._strip_subplot_suffix_of_1(prop)
    return super(BaseLayoutHierarchyType, self)._get_prop_validator(prop)