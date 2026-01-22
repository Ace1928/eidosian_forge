from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
def PeekException(self):
    """Raises an exception if the first iterated element raised."""
    if self._PopulateHead() and self.head[0][0] == 'exception':
        exception_tuple = self.head[0]
        raise six.reraise(exception_tuple[1].__class__, exception_tuple[1], exception_tuple[2])