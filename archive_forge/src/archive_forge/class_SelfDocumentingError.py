from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class SelfDocumentingError(exceptions.Error):
    """An error that uses its own docstring as its message if no message given.

  Somehow I think this was how all errors worked maybe back when this was Python
  2, and it got lost in the shuffle at some point.
  """

    def __init__(self, message):
        if message is None:
            message = self.__class__.__doc__
        super(SelfDocumentingError, self).__init__(message)