import functools
import re
import nltk.tree
def _istree(obj):
    """Predicate to check whether `obj` is a nltk.tree.Tree."""
    return isinstance(obj, nltk.tree.Tree)