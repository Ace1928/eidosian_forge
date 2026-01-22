from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class EnumClassListSerializer(ListSerializer):
    """A serializer for MultiEnumClass flags.

  This serializer simply joins the output of `EnumClassSerializer` using a
  provided separator.
  """

    def __init__(self, list_sep, **kwargs):
        """Initializes EnumClassListSerializer.

    Args:
      list_sep: String to be used as a separator when serializing
      **kwargs: Keyword arguments to the `EnumClassSerializer` used to serialize
        individual values.
    """
        super(EnumClassListSerializer, self).__init__(list_sep)
        self._element_serializer = EnumClassSerializer(**kwargs)

    def serialize(self, value):
        """See base class."""
        if isinstance(value, list):
            return self.list_sep.join((self._element_serializer.serialize(x) for x in value))
        else:
            return self._element_serializer.serialize(value)