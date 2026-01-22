from base64 import encodebytes as _bencode
from quopri import encodestring as _encodestring
def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    return enc.replace(b' ', b'=20')