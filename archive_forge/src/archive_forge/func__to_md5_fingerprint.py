import hashlib
from libcloud.utils.py3 import b, hexadigits, base64_decode_string
def _to_md5_fingerprint(data):
    hashed = hashlib.md5(data).digest()
    return ':'.join(hexadigits(hashed))