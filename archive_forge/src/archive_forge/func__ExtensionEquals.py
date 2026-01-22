from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _ExtensionEquals(self, x):
    extensions = self._ListExtensions()
    if extensions != x._ListExtensions():
        return False
    for ext in extensions:
        if ext.is_repeated:
            if self.ExtensionSize(ext) != x.ExtensionSize(ext):
                return False
            for e1, e2 in zip(self.ExtensionList(ext), x.ExtensionList(ext)):
                if e1 != e2:
                    return False
        elif self.GetExtension(ext) != x.GetExtension(ext):
            return False
    return True