import base64
import binascii
import logging
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def __find_variant(self, value):
    """Find the messages.Variant type that describes this value.

        Args:
          value: The value whose variant type is being determined.

        Returns:
          The messages.Variant value that best describes value's type,
          or None if it's a type we don't know how to handle.

        """
    if isinstance(value, bool):
        return messages.Variant.BOOL
    elif isinstance(value, six.integer_types):
        return messages.Variant.INT64
    elif isinstance(value, float):
        return messages.Variant.DOUBLE
    elif isinstance(value, six.string_types):
        return messages.Variant.STRING
    elif isinstance(value, (list, tuple)):
        variant_priority = [None, messages.Variant.INT64, messages.Variant.DOUBLE, messages.Variant.STRING]
        chosen_priority = 0
        for v in value:
            variant = self.__find_variant(v)
            try:
                priority = variant_priority.index(variant)
            except IndexError:
                priority = -1
            if priority > chosen_priority:
                chosen_priority = priority
        return variant_priority[chosen_priority]
    return None