from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def ExtensionSize(self, extension):
    """Returns the size of a repeated extension.

    Raises:
      TypeError if the extension is not repeated.
    """
    self._VerifyExtensionIdentifier(extension)
    if not extension.is_repeated:
        raise TypeError('ExtensionSize() cannot be applied to "%s", because it is not a repeated extension.' % extension.full_name)
    if extension in self._extension_fields:
        return len(self._extension_fields[extension])
    return 0