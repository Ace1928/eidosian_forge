import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def _get_masked(self, mask_key):
    s = ABNF.mask(mask_key, self.data)
    if isinstance(mask_key, six.text_type):
        mask_key = mask_key.encode('utf-8')
    return mask_key + s