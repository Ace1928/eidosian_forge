import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def _createenviron():
    if name == 'nt':

        def check_str(value):
            if not isinstance(value, str):
                raise TypeError('str expected, not %s' % type(value).__name__)
            return value
        encode = check_str
        decode = str

        def encodekey(key):
            return encode(key).upper()
        data = {}
        for key, value in environ.items():
            data[encodekey(key)] = value
    else:
        encoding = sys.getfilesystemencoding()

        def encode(value):
            if not isinstance(value, str):
                raise TypeError('str expected, not %s' % type(value).__name__)
            return value.encode(encoding, 'surrogateescape')

        def decode(value):
            return value.decode(encoding, 'surrogateescape')
        encodekey = encode
        data = environ
    return _Environ(data, encodekey, decode, encode, decode)