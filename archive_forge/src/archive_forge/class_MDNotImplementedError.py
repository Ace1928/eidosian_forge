from __future__ import annotations
from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl
class MDNotImplementedError(NotImplementedError):
    """ A NotImplementedError for multiple dispatch """