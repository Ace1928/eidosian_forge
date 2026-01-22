import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _unify_feature_values(fname, fval1, fval2, bindings, forward, trace, fail, fs_class, fpath):
    """
    Attempt to unify ``fval1`` and and ``fval2``, and return the
    resulting unified value.  The method of unification will depend on
    the types of ``fval1`` and ``fval2``:

      1. If they're both feature structures, then destructively
         unify them (see ``_destructively_unify()``.
      2. If they're both unbound variables, then alias one variable
         to the other (by setting bindings[v2]=v1).
      3. If one is an unbound variable, and the other is a value,
         then bind the unbound variable to the value.
      4. If one is a feature structure, and the other is a base value,
         then fail.
      5. If they're both base values, then unify them.  By default,
         this will succeed if they are equal, and fail otherwise.
    """
    if trace:
        _trace_unify_start(fpath, fval1, fval2)
    while id(fval1) in forward:
        fval1 = forward[id(fval1)]
    while id(fval2) in forward:
        fval2 = forward[id(fval2)]
    fvar1 = fvar2 = None
    while isinstance(fval1, Variable) and fval1 in bindings:
        fvar1 = fval1
        fval1 = bindings[fval1]
    while isinstance(fval2, Variable) and fval2 in bindings:
        fvar2 = fval2
        fval2 = bindings[fval2]
    if isinstance(fval1, fs_class) and isinstance(fval2, fs_class):
        result = _destructively_unify(fval1, fval2, bindings, forward, trace, fail, fs_class, fpath)
    elif isinstance(fval1, Variable) and isinstance(fval2, Variable):
        if fval1 != fval2:
            bindings[fval2] = fval1
        result = fval1
    elif isinstance(fval1, Variable):
        bindings[fval1] = fval2
        result = fval1
    elif isinstance(fval2, Variable):
        bindings[fval2] = fval1
        result = fval2
    elif isinstance(fval1, fs_class) or isinstance(fval2, fs_class):
        result = UnificationFailure
    else:
        if isinstance(fname, Feature):
            result = fname.unify_base_values(fval1, fval2, bindings)
        elif isinstance(fval1, CustomFeatureValue):
            result = fval1.unify(fval2)
            if isinstance(fval2, CustomFeatureValue) and result != fval2.unify(fval1):
                raise AssertionError('CustomFeatureValue objects %r and %r disagree about unification value: %r vs. %r' % (fval1, fval2, result, fval2.unify(fval1)))
        elif isinstance(fval2, CustomFeatureValue):
            result = fval2.unify(fval1)
        elif fval1 == fval2:
            result = fval1
        else:
            result = UnificationFailure
        if result is not UnificationFailure:
            if fvar1 is not None:
                bindings[fvar1] = result
                result = fvar1
            if fvar2 is not None and fvar2 != fvar1:
                bindings[fvar2] = result
                result = fvar2
    if result is UnificationFailure:
        if fail is not None:
            result = fail(fval1, fval2, fpath)
        if trace:
            _trace_unify_fail(fpath[:-1], result)
        if result is UnificationFailure:
            raise _UnificationFailureError
    if isinstance(result, fs_class):
        result = _apply_forwards(result, forward, fs_class, set())
    if trace:
        _trace_unify_succeed(fpath, result)
    if trace and isinstance(result, fs_class):
        _trace_bindings(fpath, bindings)
    return result