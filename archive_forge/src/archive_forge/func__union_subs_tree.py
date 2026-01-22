import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _union_subs_tree(tp, tvars=None, args=None):
    """ backport of Union._subs_tree """
    if tp is Union:
        return Union
    tree_args = _subs_tree(tp, tvars, args)
    tree_args = _remove_dups_flatten(tree_args)
    if len(tree_args) == 1:
        return tree_args[0]
    return (Union,) + tree_args