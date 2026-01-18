import functools
import inspect
import sys
import msgpack
import rapidjson
from ruamel import yaml
def to_msgpack(obj):
    """
    Convert Python objects (including rpcq objects) to a msgpack byte array
    :rtype: bytes
    """
    return msgpack.dumps(obj, default=_default, use_bin_type=True)