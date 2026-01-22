from ..pari import pari
import fractions
def _smith_normal_form_with_inverse(m):
    u, v, d = _internal_to_pari(m).matsnf(flag=1)
    return (_pari_to_internal(u ** (-1)), _pari_to_internal(v), _pari_to_internal(d))