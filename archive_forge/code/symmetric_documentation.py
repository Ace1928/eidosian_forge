import base64 as _base64
import logging
import os as _os
from warnings import warn as _warn
import cryptography.fernet as _fernet
import cryptography.hazmat.primitives.ciphers as _ciphers
from .errors import SymmetricCryptographyError

        :param key: The encryption key
        :param msg: Base64 encoded message to be decrypted
        :return: The decrypted message
        