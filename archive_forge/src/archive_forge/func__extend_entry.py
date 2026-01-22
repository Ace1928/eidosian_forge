import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
def _extend_entry(self, key, value):
    """Appends the `value` to the entry specified by `key`."""
    self[key].extend(value)