import json
import textwrap
@classmethod
def array_schema(cls, p, safe=False):
    if safe is True:
        msg = 'Array is not guaranteed to be safe for serialization as the dtype is unknown'
        raise UnsafeserializableException(msg)
    return {'type': 'array'}