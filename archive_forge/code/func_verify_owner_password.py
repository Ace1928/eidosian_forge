import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def verify_owner_password(R: int, password: bytes, o_value: bytes, oe_value: bytes, u_value: bytes) -> bytes:
    """
        Algorithm 3.2a Computing an encryption key.

        To understand the algorithm below, it is necessary to treat the O and U
        strings in the Encrypt dictionary as made up of three sections.
        The first 32 bytes are a hash value (explained below). The next 8 bytes
        are called the Validation Salt. The final 8 bytes are called the Key Salt.

        1. The password string is generated from Unicode input by processing the
           input string with the SASLprep (IETF RFC 4013) profile of
           stringprep (IETF RFC 3454), and then converting to a UTF-8
           representation.
        2. Truncate the UTF-8 representation to 127 bytes if it is longer than
           127 bytes.
        3. Test the password against the owner key by computing the SHA-256 hash
           of the UTF-8 password concatenated with the 8 bytes of owner
           Validation Salt, concatenated with the 48-byte U string. If the
           32-byte result matches the first 32 bytes of the O string, this is
           the owner password.
           Compute an intermediate owner key by computing the SHA-256 hash of
           the UTF-8 password concatenated with the 8 bytes of owner Key Salt,
           concatenated with the 48-byte U string. The 32-byte result is the
           key used to decrypt the 32-byte OE string using AES-256 in CBC mode
           with no padding and an initialization vector of zero.
           The 32-byte result is the file encryption key.
        4. Test the password against the user key by computing the SHA-256 hash
           of the UTF-8 password concatenated with the 8 bytes of user
           Validation Salt. If the 32 byte result matches the first 32 bytes of
           the U string, this is the user password.
           Compute an intermediate user key by computing the SHA-256 hash of the
           UTF-8 password concatenated with the 8 bytes of user Key Salt.
           The 32-byte result is the key used to decrypt the 32-byte
           UE string using AES-256 in CBC mode with no padding and an
           initialization vector of zero. The 32-byte result is the file
           encryption key.
        5. Decrypt the 16-byte Perms string using AES-256 in ECB mode with an
           initialization vector of zero and the file encryption key as the key.
           Verify that bytes 9-11 of the result are the characters ‘a’, ‘d’, ‘b’.
           Bytes 0-3 of the decrypted Perms entry, treated as a little-endian
           integer, are the user permissions.
           They should match the value in the P key.

        Args:
            R: A number specifying which revision of the standard security
                handler shall be used to interpret this dictionary
            password: The owner password
            o_value: A 32-byte string, based on both the owner and user passwords,
                that shall be used in computing the encryption key and in
                determining whether a valid owner password was entered
            oe_value:
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password and,
                if so, whether a valid user or owner password was entered.

        Returns:
            The key
        """
    password = password[:127]
    if AlgV5.calculate_hash(R, password, o_value[32:40], u_value[:48]) != o_value[:32]:
        return b''
    iv = bytes((0 for _ in range(16)))
    tmp_key = AlgV5.calculate_hash(R, password, o_value[40:48], u_value[:48])
    key = aes_cbc_decrypt(tmp_key, iv, oe_value)
    return key