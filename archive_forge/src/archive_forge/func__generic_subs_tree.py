import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _generic_subs_tree(tp, tvars=None, args=None):
    """ backport of GenericMeta._subs_tree """
    if tp.__origin__ is None:
        return tp
    tree_args = _subs_tree(tp, tvars, args)
    return (_gorg(tp),) + tuple(tree_args)