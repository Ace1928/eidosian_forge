from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
def ParseList(self, field, required=False, default=None, func=None, sort=False):
    """Parses a element out of the dictionary that is a list of items.

    Args:
      field: str, The name of the field to parse.
      required: bool, If the field must be present or not (False by default).
      default: str or dict, The value to use if a non-required field is not
        present.
      func: An optional function to call with each value in the parsed list
        before returning (if the list is not None).  It takes a single parameter
        and returns a single new value to be used instead.
      sort: bool, sort parsed list when it represents an unordered set.

    Raises:
      ParseError: If a required field is not found or if the field parsed is
        not a list.
    """
    value = self._Get(field, default, required)
    if value:
        if not isinstance(value, list):
            raise ParseError('Expected a list for field [{0}] in component [{1}]'.format(field, self.__cls))
        if func:
            value = [func(v) for v in value]
    self.__args[field] = sorted(value) if sort else value