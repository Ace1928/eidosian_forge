from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _ListExtensions(self):
    return sorted((ext for ext in self._extension_fields if not ext.is_repeated or self.ExtensionSize(ext) > 0), key=lambda item: item.number)