import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
class ViewableTree(AttrTree, Dimensioned):
    """
    A ViewableTree is an AttrTree with Viewable objects as its leaf
    nodes. It combines the tree like data structure of a tree while
    extending it with the deep indexable properties of Dimensioned
    and LabelledData objects.
    """
    group = param.String(default='ViewableTree', constant=True)
    _deep_indexable = True

    def __init__(self, items=None, identifier=None, parent=None, **kwargs):
        if items and all((isinstance(item, Dimensioned) for item in items)):
            items = self._process_items(items)
        params = {p: kwargs.pop(p) for p in list(self.param) + ['id', 'plot_id'] if p in kwargs}
        AttrTree.__init__(self, items, identifier, parent, **kwargs)
        Dimensioned.__init__(self, self.data, **params)

    @classmethod
    def _process_items(cls, vals):
        """Processes list of items assigning unique paths to each."""
        from .layout import AdjointLayout
        if type(vals) is cls:
            return vals.data
        elif isinstance(vals, (AdjointLayout, str)):
            vals = [vals]
        elif isinstance(vals, Iterable):
            vals = list(vals)
        items = []
        counts = defaultdict(lambda: 1)
        cls._unpack_paths(vals, items, counts)
        items = cls._deduplicate_items(items)
        return items

    def __setstate__(self, d):
        """
        Ensure that object does not try to reference its parent during
        unpickling.
        """
        parent = d.pop('parent', None)
        d['parent'] = None
        super(AttrTree, self).__setstate__(d)
        self.__dict__['parent'] = parent

    @classmethod
    def _deduplicate_items(cls, items):
        """Deduplicates assigned paths by incrementing numbering"""
        counter = Counter([path[:i] for path, _ in items for i in range(1, len(path) + 1)])
        if sum(counter.values()) == len(counter):
            return items
        new_items = []
        counts = defaultdict(lambda: 0)
        for path, item in items:
            if counter[path] > 1:
                path = path + (util.int_to_roman(counts[path] + 1),)
            else:
                inc = 1
                while counts[path]:
                    path = path[:-1] + (util.int_to_roman(counts[path] + inc),)
                    inc += 1
            new_items.append((path, item))
            counts[path] += 1
        return new_items

    @classmethod
    def _unpack_paths(cls, objs, items, counts):
        """
        Recursively unpacks lists and ViewableTree-like objects, accumulating
        into the supplied list of items.
        """
        if type(objs) is cls:
            objs = objs.items()
        for item in objs:
            path, obj = item if isinstance(item, tuple) else (None, item)
            if type(obj) is cls:
                cls._unpack_paths(obj, items, counts)
                continue
            new = path is None or len(path) == 1
            path = util.get_path(item) if new else path
            new_path = util.make_path_unique(path, counts, new)
            items.append((new_path, obj))

    @property
    def uniform(self):
        """Whether items in tree have uniform dimensions"""
        from .traversal import uniform
        return uniform(self)

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Concatenates values on all nodes with requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
                Whether to return the expanded values, behavior depends
                on the type of data:
                  * Columnar: If false returns unique values
                  * Geometry: If false returns scalar values per geometry
                  * Gridded: If false returns 1D coordinates
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        dimension = self.get_dimension(dimension, strict=True).name
        all_dims = self.traverse(lambda x: [d.name for d in x.dimensions()])
        if dimension in chain.from_iterable(all_dims):
            values = [el.dimension_values(dimension) for el in self if dimension in el.dimensions(label=True)]
            vals = np.concatenate(values)
            return vals if expanded else util.unique_array(vals)
        else:
            return super().dimension_values(dimension, expanded, flat)

    def __len__(self):
        return len(self.data)