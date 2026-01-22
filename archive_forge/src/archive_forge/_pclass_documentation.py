from pyrsistent._checked_types import (InvariantException, CheckedType, _restore_pickle, store_invariants)
from pyrsistent._field_common import (
from pyrsistent._transformations import transform

        Remove attribute given by name from the current instance. Raises AttributeError if the
        attribute doesn't exist.
        