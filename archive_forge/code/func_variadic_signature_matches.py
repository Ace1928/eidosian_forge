from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
from .variadic import Variadic, isvariadic
import itertools as itl
def variadic_signature_matches(types, full_signature):
    assert full_signature
    return all(variadic_signature_matches_iter(types, full_signature))