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
class PDFStandardSecurityHandlerV5(PDFStandardSecurityHandlerV4):
    supported_revisions = (5, 6)

    def init_params(self) -> None:
        super().init_params()
        self.length = 256
        self.oe = str_value(self.param['OE'])
        self.ue = str_value(self.param['UE'])
        self.o_hash = self.o[:32]
        self.o_validation_salt = self.o[32:40]
        self.o_key_salt = self.o[40:]
        self.u_hash = self.u[:32]
        self.u_validation_salt = self.u[32:40]
        self.u_key_salt = self.u[40:]
        return

    def get_cfm(self, name: str) -> Optional[Callable[[int, int, bytes], bytes]]:
        if name == 'AESV3':
            return self.decrypt_aes256
        else:
            return None

    def authenticate(self, password: str) -> Optional[bytes]:
        password_b = self._normalize_password(password)
        hash = self._password_hash(password_b, self.o_validation_salt, self.u)
        if hash == self.o_hash:
            hash = self._password_hash(password_b, self.o_key_salt, self.u)
            cipher = Cipher(algorithms.AES(hash), modes.CBC(b'\x00' * 16), backend=default_backend())
            return cipher.decryptor().update(self.oe)
        hash = self._password_hash(password_b, self.u_validation_salt)
        if hash == self.u_hash:
            hash = self._password_hash(password_b, self.u_key_salt)
            cipher = Cipher(algorithms.AES(hash), modes.CBC(b'\x00' * 16), backend=default_backend())
            return cipher.decryptor().update(self.ue)
        return None

    def _normalize_password(self, password: str) -> bytes:
        if self.r == 6:
            if not password:
                return b''
            from ._saslprep import saslprep
            password = saslprep(password)
        return password.encode('utf-8')[:127]

    def _password_hash(self, password: bytes, salt: bytes, vector: Optional[bytes]=None) -> bytes:
        """
        Compute password hash depending on revision number
        """
        if self.r == 5:
            return self._r5_password(password, salt, vector)
        return self._r6_password(password, salt[0:8], vector)

    def _r5_password(self, password: bytes, salt: bytes, vector: Optional[bytes]=None) -> bytes:
        """
        Compute the password for revision 5
        """
        hash = sha256(password)
        hash.update(salt)
        if vector is not None:
            hash.update(vector)
        return hash.digest()

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

    @staticmethod
    def _bytes_mod_3(input_bytes: bytes) -> int:
        return sum((b % 3 for b in input_bytes)) % 3

    def _aes_cbc_encrypt(self, key: bytes, iv: bytes, data: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    def decrypt_aes256(self, objid: int, genno: int, data: bytes) -> bytes:
        initialization_vector = data[:16]
        ciphertext = data[16:]
        assert self.key is not None
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(initialization_vector), backend=default_backend())
        return cipher.decryptor().update(ciphertext)