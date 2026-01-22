from __future__ import annotations
import copy
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import (
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.datatree_.datatree.common import TreeAttrAccessMixin
from xarray.datatree_.datatree.formatting import datatree_repr
from xarray.datatree_.datatree.formatting_html import (
from xarray.datatree_.datatree.mapping import (
from xarray.datatree_.datatree.ops import (
from xarray.datatree_.datatree.render import RenderTree
@classmethod
def _from_node(cls, wrapping_node: DataTree) -> DatasetView:
    """Constructor, using dataset attributes from wrapping node"""
    obj: DatasetView = object.__new__(cls)
    obj._variables = wrapping_node._variables
    obj._coord_names = wrapping_node._coord_names
    obj._dims = wrapping_node._dims
    obj._indexes = wrapping_node._indexes
    obj._attrs = wrapping_node._attrs
    obj._close = wrapping_node._close
    obj._encoding = wrapping_node._encoding
    return obj