import secrets
from Crypto import __version__
from Crypto.Cipher import AES, ARC4
from Crypto.Util.Padding import pad
from pypdf._crypt_providers._base import CryptBase
class CryptAES(CryptBase):

    def __init__(self, key: bytes) -> None:
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        iv = secrets.token_bytes(16)
        data = pad(data, 16)
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + aes.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        iv = data[:16]
        data = data[16:]
        if not data:
            return data
        if len(data) % 16 != 0:
            data = pad(data, 16)
        aes = AES.new(self.key, AES.MODE_CBC, iv)
        d = aes.decrypt(data)
        return d[:-d[-1]]