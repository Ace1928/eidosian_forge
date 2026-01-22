import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
class SliceableDataDict(MutableMapping):
    """Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.

    Parameters
    ----------
    \\*args :
    \\*\\*kwargs :
        Positional and keyword arguments, passed straight through the ``dict``
        constructor.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        try:
            return self.store[key]
        except (KeyError, TypeError, IndexError):
            pass
        idx = key
        new_dict = type(self)()
        try:
            for k, v in self.items():
                new_dict[k] = v[idx]
        except (TypeError, ValueError, IndexError):
            pass
        else:
            return new_dict
        return self.store[key]

    def __contains__(self, key):
        return key in self.store

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)