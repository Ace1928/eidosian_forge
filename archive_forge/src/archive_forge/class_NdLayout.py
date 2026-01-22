import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
class NdLayout(Layoutable, UniformNdMapping):
    """
    NdLayout is a UniformNdMapping providing an n-dimensional
    data structure to display the contained Elements and containers
    in a layout. Using the cols method the NdLayout can be rearranged
    with the desired number of columns.
    """
    data_type = (ViewableElement, AdjointLayout, UniformNdMapping)

    def __init__(self, initial_items=None, kdims=None, **params):
        self._max_cols = 4
        self._style = None
        super().__init__(initial_items=initial_items, kdims=kdims, **params)

    @property
    def uniform(self):
        return traversal.uniform(self)

    @property
    def shape(self):
        """Tuple indicating the number of rows and columns in the NdLayout."""
        num = len(self.keys())
        if num <= self._max_cols:
            return (1, num)
        nrows = num // self._max_cols
        last_row_cols = num % self._max_cols
        return (nrows + (1 if last_row_cols else 0), min(num, self._max_cols))

    def grid_items(self):
        """
        Compute a dict of {(row,column): (key, value)} elements from the
        current set of items and specified number of columns.
        """
        if list(self.keys()) == []:
            return {}
        cols = self._max_cols
        return {(idx // cols, idx % cols): (key, item) for idx, (key, item) in enumerate(self.data.items())}

    def cols(self, ncols):
        """Sets the maximum number of columns in the NdLayout.

        Any items beyond the set number of cols will flow onto a new
        row. The number of columns control the indexing and display
        semantics of the NdLayout.

        Args:
            ncols (int): Number of columns to set on the NdLayout
        """
        self._max_cols = ncols
        return self

    @property
    def last(self):
        """
        Returns another NdLayout constituted of the last views of the
        individual elements (if they are maps).
        """
        last_items = []
        for k, v in self.items():
            if isinstance(v, NdMapping):
                item = (k, v.clone((v.last_key, v.last)))
            elif isinstance(v, AdjointLayout):
                item = (k, v.last)
            else:
                item = (k, v)
            last_items.append(item)
        return self.clone(last_items)

    def clone(self, *args, **overrides):
        """Clones the NdLayout, overriding data and parameters.

        Args:
            data: New data replacing the existing data
            shared_data (bool, optional): Whether to use existing data
            new_type (optional): Type to cast object to
            *args: Additional arguments to pass to constructor
            **overrides: New keyword arguments to pass to constructor

        Returns:
            Cloned NdLayout object
        """
        clone = super().clone(*args, **overrides)
        clone._max_cols = self._max_cols
        clone.id = self.id
        return clone