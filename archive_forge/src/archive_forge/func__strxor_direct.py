from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
def _strxor_direct(term1, term2, result):
    """Very fast XOR - check conditions!"""
    _raw_strxor.strxor(term1, term2, result, c_size_t(len(term1)))