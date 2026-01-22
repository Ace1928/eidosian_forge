import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
class DecryptionError(CryptoError):
    """Raised when decryption fails."""