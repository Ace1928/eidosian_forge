import re
from binascii import a2b_base64, b2a_base64, hexlify, unhexlify
from Cryptodome.Hash import MD5
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Cipher import DES, DES3, AES
from Cryptodome.Protocol.KDF import PBKDF1
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import tobytes, tostr
def _EVP_BytesToKey(data, salt, key_len):
    d = [b'']
    m = (key_len + 15) // 16
    for _ in range(m):
        nd = MD5.new(d[-1] + data + salt).digest()
        d.append(nd)
    return b''.join(d)[:key_len]