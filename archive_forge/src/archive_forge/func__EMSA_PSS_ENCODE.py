from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
def _EMSA_PSS_ENCODE(mhash, emBits, randFunc, mgf, sLen):
    """
    Implement the ``EMSA-PSS-ENCODE`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.1.1).

    The original ``EMSA-PSS-ENCODE`` actually accepts the message ``M``
    as input, and hash it internally. Here, we expect that the message
    has already been hashed instead.

    :Parameters:
      mhash : hash object
        The hash object that holds the digest of the message being signed.
      emBits : int
        Maximum length of the final encoding, in bits.
      randFunc : callable
        An RNG function that accepts as only parameter an int, and returns
        a string of random bytes, to be used as salt.
      mgf : callable
        A mask generation function that accepts two parameters: a string to
        use as seed, and the lenth of the mask to generate, in bytes.
      sLen : int
        Length of the salt, in bytes.

    :Return: An ``emLen`` byte long string that encodes the hash
      (with ``emLen = \\ceil(emBits/8)``).

    :Raise ValueError:
        When digest or salt length are too big.
    """
    emLen = ceil_div(emBits, 8)
    lmask = 0
    for i in iter_range(8 * emLen - emBits):
        lmask = lmask >> 1 | 128
    if emLen < mhash.digest_size + sLen + 2:
        raise ValueError('Digest or salt length are too long for given key size.')
    salt = randFunc(sLen)
    m_prime = bchr(0) * 8 + mhash.digest() + salt
    h = mhash.new()
    h.update(m_prime)
    ps = bchr(0) * (emLen - sLen - mhash.digest_size - 2)
    db = ps + bchr(1) + salt
    dbMask = mgf(h.digest(), emLen - mhash.digest_size - 1)
    maskedDB = strxor(db, dbMask)
    maskedDB = bchr(bord(maskedDB[0]) & ~lmask) + maskedDB[1:]
    em = maskedDB + h.digest() + bchr(188)
    return em