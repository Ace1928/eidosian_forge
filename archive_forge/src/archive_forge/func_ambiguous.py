from .utils import _toposort, groupby
def ambiguous(a, b):
    """ A is consistent with B but neither is strictly more specific """
    return consistent(a, b) and (not (supercedes(a, b) or supercedes(b, a)))