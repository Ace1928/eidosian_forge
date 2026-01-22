import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
class PerArraySequenceDict(PerArrayDict):
    """Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.  The elements must also be :class:`ArraySequence`.

    In addition, it makes sure the amount of data contained in those array
    sequences matches the number of elements given at the instantiation
    of the instance.
    """

    def __setitem__(self, key, value):
        value = ArraySequence(value)
        if 0 < self.n_rows != value.total_nb_rows:
            msg = f'The number of values ({value.total_nb_rows}) should match ({self.n_rows}).'
            raise ValueError(msg)
        self.store[key] = value

    def _extend_entry(self, key, value):
        """Appends the `value` to the entry specified by `key`."""
        self[key].extend(value)