from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
from absl._collections_abc import abc
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers
import six
def _serialize_value_for_xml(self, value):
    """See base class."""
    if value is not None:
        value_serialized = self.serializer.serialize(value)
    else:
        value_serialized = ''
    return value_serialized