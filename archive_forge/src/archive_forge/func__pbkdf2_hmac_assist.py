from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def _pbkdf2_hmac_assist(inner, outer, first_digest, iterations):
    """Compute the expensive inner loop in PBKDF-HMAC."""
    assert iterations > 0
    bfr = create_string_buffer(len(first_digest))
    result = _raw_sha512_lib.SHA512_pbkdf2_hmac_assist(inner._state.get(), outer._state.get(), first_digest, bfr, c_size_t(iterations), c_size_t(len(first_digest)))
    if result:
        raise ValueError('Error %d with PBKDF2-HMAC assist for SHA512' % result)
    return get_raw_buffer(bfr)