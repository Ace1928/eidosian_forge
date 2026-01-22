import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def _is_encryption_object(obj: PdfObject) -> bool:
    return isinstance(obj, (ByteStringObject, TextStringObject, StreamObject, ArrayObject, DictionaryObject))