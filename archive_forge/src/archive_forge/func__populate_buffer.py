from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
from googlecloudsdk.core import exceptions
def _populate_buffer(self, num_elements=1):
    while len(self._buffer) < num_elements:
        try:
            self._buffer.append(next(self._iterator))
        except StopIteration:
            break
        except Exception as e:
            self._buffer.append(BufferedException(exception=e, stack_trace=sys.exc_info()[2]))