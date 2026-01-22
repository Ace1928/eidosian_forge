from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _make_seq_field_type(checked_class, item_type, item_invariant):
    """Create a subclass of the given checked class with the given item type."""
    type_ = _seq_field_types.get((checked_class, item_type))
    if type_ is not None:
        return type_

    class TheType(checked_class):
        __type__ = item_type
        __invariant__ = item_invariant

        def __reduce__(self):
            return (_restore_seq_field_pickle, (checked_class, item_type, list(self)))
    suffix = SEQ_FIELD_TYPE_SUFFIXES[checked_class]
    TheType.__name__ = _types_to_names(TheType._checked_types) + suffix
    _seq_field_types[checked_class, item_type] = TheType
    return TheType