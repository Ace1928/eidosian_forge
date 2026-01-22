from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
class FilterRefBase(Mapping):
    """Base class for referring to an HDF5 and describing its options

    Your subclass must define filter_id, and may define a filter_options tuple.
    """
    filter_id = None
    filter_options = ()

    @property
    def _kwargs(self):
        return {'compression': self.filter_id, 'compression_opts': self.filter_options}

    def __hash__(self):
        return hash((self.filter_id, self.filter_options))

    def __eq__(self, other):
        return isinstance(other, FilterRefBase) and self.filter_id == other.filter_id and (self.filter_options == other.filter_options)

    def __len__(self):
        return len(self._kwargs)

    def __iter__(self):
        return iter(self._kwargs)

    def __getitem__(self, item):
        return self._kwargs[item]