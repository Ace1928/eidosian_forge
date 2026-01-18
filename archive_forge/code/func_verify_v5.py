import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def verify_v5(self, password: bytes) -> Tuple[bytes, PasswordType]:
    key = AlgV5.verify_owner_password(self.R, password, self.values.O, self.values.OE, self.values.U)
    rc = PasswordType.OWNER_PASSWORD
    if not key:
        key = AlgV5.verify_user_password(self.R, password, self.values.U, self.values.UE)
        rc = PasswordType.USER_PASSWORD
    if not key:
        return (b'', PasswordType.NOT_DECRYPTED)
    if not AlgV5.verify_perms(key, self.values.Perms, self.P, self.EncryptMetadata):
        logger_warning("ignore '/Perms' verify failed", __name__)
    return (key, rc)