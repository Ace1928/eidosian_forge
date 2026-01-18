from __future__ import annotations
from monty.json import MontyDecoder, MontyEncoder

    For use with msgpack.unpackb(dict, object_hook=object_hook.).  Supports
    Monty's as_dict protocol, numpy arrays and datetime.
    