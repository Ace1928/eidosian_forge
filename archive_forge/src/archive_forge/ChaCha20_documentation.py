from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
Seek to a certain position in the key stream.

        Args:
          position (integer):
            The absolute position within the key stream, in bytes.
        