from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, tobytes
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
Return a new instance of a BLAKE2s hash object.
        See :func:`new`.
        