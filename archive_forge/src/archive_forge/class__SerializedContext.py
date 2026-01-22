import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class _SerializedContext(SupportsJSON):
    """Internal object for a single SerializableByKey key-to-object mapping.

    This is a private type used in contextual serialization. Its deserialization
    is context-dependent, and is not expected to match the original; in other
    words, `cls._from_json_dict_(obj._json_dict_())` does not return
    the original `obj` for this type.
    """

    def __init__(self, obj: SerializableByKey, uid: int):
        self.key = uid
        self.obj = obj

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['key', 'obj'])

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        raise TypeError(f'Internal error: {cls} should never deserialize with _from_json_dict_.')

    @classmethod
    def update_context(cls, context_map, key, obj, **kwargs):
        context_map.update({key: obj})