from .utils import _toposort, groupby
def ambiguities(signatures):
    """ All signature pairs such that A is ambiguous with B """
    signatures = list(map(tuple, signatures))
    return {(a, b) for a in signatures for b in signatures if hash(a) < hash(b) and ambiguous(a, b) and (not any((supercedes(c, a) and supercedes(c, b) for c in signatures)))}