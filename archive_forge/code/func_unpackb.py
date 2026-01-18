import functools
import catalogue
from ._version import version
from .exceptions import *
from ._packer import Packer as _Packer
from ._unpacker import unpackb as _unpackb
from ._unpacker import unpack as _unpack
from ._unpacker import Unpacker as _Unpacker
from ._ext_type import ExtType
from ._msgpack_numpy import encode_numpy as _encode_numpy
from ._msgpack_numpy import decode_numpy as _decode_numpy
def unpackb(packed, **kwargs):
    """
    Unpack a packed object.
    """
    if 'object_pairs_hook' not in kwargs:
        object_hook = kwargs.get('object_hook')
        for decoder in msgpack_decoders.get_all().values():
            object_hook = functools.partial(decoder, chain=object_hook)
        kwargs['object_hook'] = object_hook
    return _unpackb(packed, **kwargs)