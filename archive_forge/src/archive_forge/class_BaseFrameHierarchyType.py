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
class BaseFrameHierarchyType(BasePlotlyType):
    """
    Base class for all types in the trace hierarchy
    """

    def __init__(self, plotly_name, **kwargs):
        super(BaseFrameHierarchyType, self).__init__(plotly_name, **kwargs)

    def _send_prop_set(self, prop_path_str, val):
        pass

    def _restyle_child(self, child, key_path_str, val):
        pass

    def on_change(self, callback, *args):
        raise NotImplementedError('Change callbacks are not supported on Frames')

    def _get_child_props(self, child):
        """
        Return the properties dict for a child trace or child layout

        Note: this method must match the name/signature of one on
        BasePlotlyType

        Parameters
        ----------
        child : BaseTraceType | BaseLayoutType

        Returns
        -------
        dict
        """
        try:
            trace_index = BaseFigure._index_is(self.data, child)
        except ValueError:
            trace_index = None
        if trace_index is not None:
            if 'data' in self._props:
                return self._props['data'][trace_index]
            else:
                return None
        elif child is self.layout:
            return self._props.get('layout', None)
        else:
            raise ValueError('Unrecognized child: %s' % child)