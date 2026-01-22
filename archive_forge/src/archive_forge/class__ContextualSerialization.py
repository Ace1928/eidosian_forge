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
class _ContextualSerialization(SupportsJSON):
    """Internal object for serializing an object with its context.

    This is a private type used in contextual serialization. Its deserialization
    is context-dependent, and is not expected to match the original; in other
    words, `cls._from_json_dict_(obj._json_dict_())` does not return
    the original `obj` for this type.
    """

    def __init__(self, obj: Any):
        self.object_dag = []
        context = []
        for sbk in get_serializable_by_keys(obj):
            if sbk not in context:
                context.append(sbk)
                new_sc = _SerializedContext(sbk, len(context))
                self.object_dag.append(new_sc)
        self.object_dag += [obj]

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['object_dag'])

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        raise TypeError(f'Internal error: {cls} should never deserialize with _from_json_dict_.')

    @classmethod
    def deserialize_with_context(cls, object_dag, **kwargs):
        return object_dag[-1]