import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _destructively_unify(fstruct1, fstruct2, bindings, forward, trace, fail, fs_class, path):
    """
    Attempt to unify ``fstruct1`` and ``fstruct2`` by modifying them
    in-place.  If the unification succeeds, then ``fstruct1`` will
    contain the unified value, the value of ``fstruct2`` is undefined,
    and forward[id(fstruct2)] is set to fstruct1.  If the unification
    fails, then a _UnificationFailureError is raised, and the
    values of ``fstruct1`` and ``fstruct2`` are undefined.

    :param bindings: A dictionary mapping variables to values.
    :param forward: A dictionary mapping feature structures ids
        to replacement structures.  When two feature structures
        are merged, a mapping from one to the other will be added
        to the forward dictionary; and changes will be made only
        to the target of the forward dictionary.
        ``_destructively_unify`` will always 'follow' any links
        in the forward dictionary for fstruct1 and fstruct2 before
        actually unifying them.
    :param trace: If true, generate trace output
    :param path: The feature path that led us to this unification
        step.  Used for trace output.
    """
    if fstruct1 is fstruct2:
        if trace:
            _trace_unify_identity(path, fstruct1)
        return fstruct1
    forward[id(fstruct2)] = fstruct1
    if _is_mapping(fstruct1) and _is_mapping(fstruct2):
        for fname in fstruct1:
            if getattr(fname, 'default', None) is not None:
                fstruct2.setdefault(fname, fname.default)
        for fname in fstruct2:
            if getattr(fname, 'default', None) is not None:
                fstruct1.setdefault(fname, fname.default)
        for fname, fval2 in sorted(fstruct2.items()):
            if fname in fstruct1:
                fstruct1[fname] = _unify_feature_values(fname, fstruct1[fname], fval2, bindings, forward, trace, fail, fs_class, path + (fname,))
            else:
                fstruct1[fname] = fval2
        return fstruct1
    elif _is_sequence(fstruct1) and _is_sequence(fstruct2):
        if len(fstruct1) != len(fstruct2):
            return UnificationFailure
        for findex in range(len(fstruct1)):
            fstruct1[findex] = _unify_feature_values(findex, fstruct1[findex], fstruct2[findex], bindings, forward, trace, fail, fs_class, path + (findex,))
        return fstruct1
    elif (_is_sequence(fstruct1) or _is_mapping(fstruct1)) and (_is_sequence(fstruct2) or _is_mapping(fstruct2)):
        return UnificationFailure
    raise TypeError('Expected mappings or sequences')