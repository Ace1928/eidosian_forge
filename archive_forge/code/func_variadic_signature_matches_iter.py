from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
from .variadic import Variadic, isvariadic
import itertools as itl
def variadic_signature_matches_iter(types, full_signature):
    """Check if a set of input types matches a variadic signature.
    Notes
    -----
    The algorithm is as follows:
    Initialize the current signature to the first in the sequence
    For each type in `types`:
        If the current signature is variadic
            If the type matches the signature
                yield True
            Else
                Try to get the next signature
                If no signatures are left we can't possibly have a match
                    so yield False
        Else
            yield True if the type matches the current signature
            Get the next signature
    """
    sigiter = iter(full_signature)
    sig = next(sigiter)
    for typ in types:
        matches = issubclass(typ, sig)
        yield matches
        if not isvariadic(sig):
            sig = next(sigiter)
    else:
        try:
            sig = next(sigiter)
        except StopIteration:
            assert isvariadic(sig)
            yield True
        else:
            yield False