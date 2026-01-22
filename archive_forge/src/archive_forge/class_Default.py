import base64 as _base64
import logging
import os as _os
from warnings import warn as _warn
import cryptography.fernet as _fernet
import cryptography.hazmat.primitives.ciphers as _ciphers
from .errors import SymmetricCryptographyError
class Default(Fernet):
    """Default class is saml2.cryptography.symmetric.Fernet"""