from ..pari import pari
import fractions
def _inner_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([e1 * e2 for e1, e2 in zip(v1, v2)])