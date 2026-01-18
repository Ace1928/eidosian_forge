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
def to_json_gzip(obj: Any, file_or_fn: Union[None, IO, pathlib.Path, str]=None, *, indent: int=2, cls: Type[json.JSONEncoder]=CirqEncoder) -> Optional[bytes]:
    """Write a gzipped JSON file containing a representation of obj.

    The object may be a cirq object or have data members that are cirq
    objects which implement the SupportsJSON protocol.

    Args:
        obj: An object which can be serialized to a JSON representation.
        file_or_fn: A filename (if a string or `pathlib.Path`) to write to, or
            an IO object (such as a file or buffer) to write to, or `None` to
            indicate that the method should return the JSON text as its result.
            Defaults to `None`.
        indent: Pretty-print the resulting file with this indent level.
            Passed to json.dump.
        cls: Passed to json.dump; the default value of CirqEncoder
            enables the serialization of Cirq objects which implement
            the SupportsJSON protocol. To support serialization of 3rd
            party classes, prefer adding the _json_dict_ magic method
            to your classes rather than overriding this default.
    """
    json_str = to_json(obj, indent=indent, cls=cls)
    if isinstance(file_or_fn, (str, pathlib.Path)):
        with gzip.open(file_or_fn, 'wt', encoding='utf-8') as actually_a_file:
            actually_a_file.write(json_str)
            return None
    gzip_data = gzip.compress(bytes(json_str, encoding='utf-8'))
    if file_or_fn is None:
        return gzip_data
    file_or_fn.write(gzip_data)
    return None