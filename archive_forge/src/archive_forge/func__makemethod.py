from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def _makemethod(name):
    return lambda self, *pos, **kw: self._savedmethods[name](*pos, **kw)