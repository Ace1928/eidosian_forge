import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock

        mask or unmask data. Just do xor for each byte

        mask_key: 4 byte string(byte).

        data: data to mask/unmask.
        