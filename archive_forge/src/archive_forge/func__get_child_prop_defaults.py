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
def _get_child_prop_defaults(self, child):
    """
        Return default properties dict for child

        Parameters
        ----------
        child : BasePlotlyType

        Returns
        -------
        dict
        """
    if self._prop_defaults is None:
        return None
    elif child.plotly_name in self._compound_props:
        return self._prop_defaults.get(child.plotly_name, None)
    elif child.plotly_name in self._compound_array_props:
        children = self._compound_array_props[child.plotly_name]
        child_ind = BaseFigure._index_is(children, child)
        assert child_ind is not None
        children_props = self._prop_defaults.get(child.plotly_name, None)
        return children_props[child_ind] if children_props is not None and len(children_props) > child_ind else None
    else:
        raise ValueError('Invalid child with name: %s' % child.plotly_name)