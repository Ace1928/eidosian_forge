import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def unify(t1, t2):
    """Unify the two types t1 and t2.

    Makes the types t1 and t2 the same.

    Args:
        t1: The first type to be made equivalent
        t2: The second type to be be equivalent

    Returns:
        None

    Raises:
        InferenceError: Raised if the types cannot be unified.
    """
    a = prune(t1)
    b = prune(t2)
    if isinstance(a, TypeVariable):
        if a != b:
            if occurs_in_type(a, b):
                raise InferenceError('recursive unification')
            a.instance = b
    elif isinstance(b, TypeVariable):
        unify(b, a)
    elif isinstance(a, TypeOperator) and a.name == 'any':
        return
    elif isinstance(b, TypeOperator) and b.name == 'any':
        return
    elif isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
        if len(a.types) != len(b.types):
            raise InferenceError('Type length differ')
        elif a.name != b.name:
            raise InferenceError('Type name differ')
        try:
            for p, q in zip(a.types, b.types):
                unify(p, q)
        except InferenceError:
            raise
    elif isinstance(a, MultiType) and isinstance(b, MultiType):
        if len(a.types) != len(b.types):
            raise InferenceError('Type lenght differ')
        for p, q in zip(a.types, b.types):
            unify(p, q)
    elif isinstance(b, MultiType):
        return unify(b, a)
    elif isinstance(a, MultiType):
        types = []
        for t in a.types:
            try:
                t_clone = fresh(t, {})
                b_clone = fresh(b, {})
                unify(t_clone, b_clone)
                types.append(t)
            except InferenceError:
                pass
        if types:
            if len(types) == 1:
                unify(clone(types[0]), b)
            else:

                def try_unify(t, ts):
                    if isinstance(t, TypeVariable):
                        return
                    if any((isinstance(tp, TypeVariable) for tp in ts)):
                        return
                    if any((len(tp.types) != len(t.types) for tp in ts)):
                        return
                    for i, tt in enumerate(t.types):
                        its = [prune(tp.types[i]) for tp in ts]
                        if any((isinstance(it, TypeVariable) for it in its)):
                            continue
                        it0 = its[0]
                        it0ntypes = len(it0.types)
                        if all((it.name == it0.name and len(it.types) == it0ntypes for it in its)):
                            ntypes = [TypeVariable() for _ in range(it0ntypes)]
                            new_tt = TypeOperator(it0.name, ntypes)
                            new_tt.__class__ = it0.__class__
                            unify(tt, new_tt)
                            try_unify(prune(tt), [prune(it) for it in its])
                try_unify(b, types)
        else:
            raise InferenceError('No overload')
    else:
        raise RuntimeError('Not unified {} and {}'.format(type(a), type(b)))