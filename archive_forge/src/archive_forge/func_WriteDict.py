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
def WriteDict(self, field, func=None):
    """Writes the given dict field to the dictionary.

    This gets the value of the attribute named field from self, and writes that
    to the dictionary.  The field is not written if the value is not set.

    Args:
      field: str, The field name.
      func: An optional function to call on each value in the dict before
        writing it to the dictionary.
    """

    def DictMapper(values):
        return dict(((k, func(v)) for k, v in six.iteritems(values)))
    dict_func = DictMapper if func else None
    self.Write(field, func=dict_func)