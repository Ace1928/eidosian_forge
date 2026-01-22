from __future__ import absolute_import
import sys
from googlecloudsdk.third_party.appengine._internal import six_subset
def CheckSuccess(self):
    """If there was an exception, raise it now.

    Raises:
      Exception of the API call or the callback, if any.
    """
    if self.exception and self._traceback:
        six_subset.reraise(self.exception.__class__, self.exception, self._traceback)
    elif self.exception:
        raise self.exception