from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
def _validate_type_encoder(self, codec: _Codec) -> None:
    from bson import _BUILT_IN_TYPES
    for pytype in _BUILT_IN_TYPES:
        if issubclass(cast(TypeCodec, codec).python_type, pytype):
            err_msg = f'TypeEncoders cannot change how built-in types are encoded (encoder {codec} transforms type {pytype})'
            raise TypeError(err_msg)