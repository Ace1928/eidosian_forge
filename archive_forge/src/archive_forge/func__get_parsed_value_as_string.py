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
def _get_parsed_value_as_string(self, value):
    """Returns parsed flag value as string."""
    if value is None:
        return None
    if self.serializer:
        return repr(self.serializer.serialize(value))
    if self.boolean:
        if value:
            return repr('true')
        else:
            return repr('false')
    return repr(_helpers.str_or_unicode(value))