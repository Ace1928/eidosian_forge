import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
def _r6_password(self, password: bytes, salt: bytes, vector: Optional[bytes]=None) -> bytes:
    """
        Compute the password for revision 6
        """
    initial_hash = sha256(password)
    initial_hash.update(salt)
    if vector is not None:
        initial_hash.update(vector)
    k = initial_hash.digest()
    hashes = (sha256, sha384, sha512)
    round_no = last_byte_val = 0
    while round_no < 64 or last_byte_val > round_no - 32:
        k1 = (password + k + (vector or b'')) * 64
        e = self._aes_cbc_encrypt(key=k[:16], iv=k[16:32], data=k1)
        next_hash = hashes[self._bytes_mod_3(e[:16])]
        k = next_hash(e).digest()
        last_byte_val = e[len(e) - 1]
        round_no += 1
    return k[:32]