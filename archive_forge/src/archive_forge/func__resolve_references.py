import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _resolve_references(self, references):
    """Return the resolved key references for references.

        References are resolved by looking up the location of the key in the
        _keys_by_offset map and substituting the key name, preserving ordering.

        :param references: An iterable of iterables of key locations. e.g.
            [[123, 456], [123]]
        :return: A tuple of tuples of keys.
        """
    node_refs = []
    for ref_list in references:
        node_refs.append(tuple([self._keys_by_offset[ref][0] for ref in ref_list]))
    return tuple(node_refs)