from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
def MGF1(mgfSeed, maskLen, hash_gen):
    """Mask Generation Function, described in `B.2.1 of RFC8017
    <https://tools.ietf.org/html/rfc8017>`_.

    :param mfgSeed:
        seed from which the mask is generated
    :type mfgSeed: byte string

    :param maskLen:
        intended length in bytes of the mask
    :type maskLen: integer

    :param hash_gen:
        A module or a hash object from :mod:`Cryptodome.Hash`
    :type hash_object:

    :return: the mask, as a *byte string*
    """
    T = b''
    for counter in iter_range(ceil_div(maskLen, hash_gen.digest_size)):
        c = long_to_bytes(counter, 4)
        hobj = hash_gen.new()
        hobj.update(mgfSeed + c)
        T = T + hobj.digest()
    assert len(T) >= maskLen
    return T[:maskLen]