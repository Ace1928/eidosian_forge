from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _make_pmap_field_type(key_type, value_type):
    """Create a subclass of CheckedPMap with the given key and value types."""
    type_ = _pmap_field_types.get((key_type, value_type))
    if type_ is not None:
        return type_

    class TheMap(CheckedPMap):
        __key_type__ = key_type
        __value_type__ = value_type

        def __reduce__(self):
            return (_restore_pmap_field_pickle, (self.__key_type__, self.__value_type__, dict(self)))
    TheMap.__name__ = '{0}To{1}PMap'.format(_types_to_names(TheMap._checked_key_types), _types_to_names(TheMap._checked_value_types))
    _pmap_field_types[key_type, value_type] = TheMap
    return TheMap