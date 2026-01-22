from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _VerifyExtensionIdentifier(self, extension):
    if extension.containing_cls != self.__class__:
        raise TypeError('Containing type of %s is %s, but not %s.' % (extension.full_name, extension.containing_cls.__name__, self.__class__.__name__))