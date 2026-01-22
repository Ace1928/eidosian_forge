from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
class PTypeError(TypeError):
    """
    Raised when trying to assign a value with a type that doesn't match the declared type.

    Attributes:
    source_class -- The class of the record
    field -- Field name
    expected_types  -- Types allowed for the field
    actual_type -- The non matching type
    """

    def __init__(self, source_class, field, expected_types, actual_type, *args, **kwargs):
        super(PTypeError, self).__init__(*args, **kwargs)
        self.source_class = source_class
        self.field = field
        self.expected_types = expected_types
        self.actual_type = actual_type