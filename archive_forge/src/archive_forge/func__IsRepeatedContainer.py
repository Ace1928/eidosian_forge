import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
def _IsRepeatedContainer(value):
    if isinstance(value, six.string_types):
        return False
    try:
        iter(value)
        return True
    except TypeError:
        return False