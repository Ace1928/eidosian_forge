import cgi
import copy
import sys
from collections.abc import MutableMapping as DictMixin
def _encode_key(self, key):
    if self.decode_keys:
        try:
            key = key.encode(self.encoding, self.errors)
        except AttributeError:
            pass
    return key