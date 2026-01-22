import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
def _get_indices_from_slice(self, row, column):
    """convert a two-dimensional slice to a list of rows and column indices"""
    if isinstance(row, slice):
        start, stop, stride = row.indices(self.n_rows)
        rows = range(start, stop, stride)
    else:
        rows = [row]
    if isinstance(column, slice):
        start, stop, stride = column.indices(self.n_columns)
        columns = range(start, stop, stride)
    else:
        columns = [column]
    return (rows, columns)