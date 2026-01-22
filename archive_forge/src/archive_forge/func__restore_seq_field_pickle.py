from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _restore_seq_field_pickle(checked_class, item_type, data):
    """Unpickling function for auto-generated PVec/PSet field types."""
    type_ = _seq_field_types[checked_class, item_type]
    return _restore_pickle(type_, data)