from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
class TheType(checked_class):
    __type__ = item_type
    __invariant__ = item_invariant

    def __reduce__(self):
        return (_restore_seq_field_pickle, (checked_class, item_type, list(self)))