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
def append_trace(self, trace, row, col):
    """
        Add a trace to the figure bound to axes at the specified row,
        col index.

        A row, col index grid is generated for figures created with
        plotly.tools.make_subplots, and can be viewed with the `print_grid`
        method

        Parameters
        ----------
        trace
            The data trace to be bound
        row: int
            Subplot row index (see Figure.print_grid)
        col: int
            Subplot column index (see Figure.print_grid)

        Examples
        --------

        >>> from plotly import tools
        >>> import plotly.graph_objs as go
        >>> # stack two subplots vertically
        >>> fig = tools.make_subplots(rows=2)

        This is the format of your plot grid:
        [ (1,1) x1,y1 ]
        [ (2,1) x2,y2 ]

        >>> fig.append_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=1)
        >>> fig.append_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=2, col=1)
        """
    warnings.warn('The append_trace method is deprecated and will be removed in a future version.\nPlease use the add_trace method with the row and col parameters.\n', DeprecationWarning)
    self.add_trace(trace=trace, row=row, col=col)