import re
from binascii import a2b_base64, b2a_base64, hexlify, unhexlify
from Cryptodome.Hash import MD5
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Cipher import DES, DES3, AES
from Cryptodome.Protocol.KDF import PBKDF1
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import tobytes, tostr
Decode a PEM block into binary.

    Args:
      pem_data (string):
        The PEM block.
      passphrase (byte string):
        If given and the PEM block is encrypted,
        the key will be derived from the passphrase.

    Returns:
      A tuple with the binary data, the marker string, and a boolean to
      indicate if decryption was performed.

    Raises:
      ValueError: if decoding fails, if the PEM file is encrypted and no passphrase has
                  been provided or if the passphrase is incorrect.
    