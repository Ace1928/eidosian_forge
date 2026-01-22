from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
class PluralityCheckableIterator(six.Iterator):
    """Iterator wrapper class.

    Allows you to check whether the wrapped iterator is empty and
    whether it has more than 1 element. This iterator accepts three types of
    values from the iterator it wraps:
      1. A yielded element (this is the normal case).
      2. A raised exception, which will be buffered and re-raised when it
         is reached in this iterator.
      3. A yielded tuple of (exception, stack trace), which will be buffered
         and raised with it is reached in this iterator.
  """

    def __init__(self, it):
        self.orig_iterator = it
        self.base_iterator = None
        self.head = []
        self.underlying_iter_empty = False

    def _PopulateHead(self, num_elements=1):
        """Populates self.head from the underlying iterator.

    Args:
      num_elements: Populate until self.head contains this many
          elements (or until the underlying iterator runs out).

    Returns:
      Number of elements at self.head after execution complete.
    """
        while not self.underlying_iter_empty and len(self.head) < num_elements:
            try:
                if not self.base_iterator:
                    self.base_iterator = iter(self.orig_iterator)
                e = next(self.base_iterator)
                self.underlying_iter_empty = False
                if isinstance(e, tuple) and isinstance(e[0], Exception):
                    self.head.append(('exception', e[0], e[1]))
                else:
                    self.head.append(('element', e))
            except StopIteration:
                self.underlying_iter_empty = True
            except Exception as e:
                self.head.append(('exception', e, sys.exc_info()[2]))
        return len(self.head)

    def __iter__(self):
        return self

    def __next__(self):
        if self._PopulateHead():
            item_tuple = self.head.pop(0)
            if item_tuple[0] == 'element':
                return item_tuple[1]
            else:
                raise six.reraise(item_tuple[1].__class__, item_tuple[1], item_tuple[2])
        raise StopIteration()

    def IsEmpty(self):
        return not self._PopulateHead()

    def HasPlurality(self):
        return self._PopulateHead(num_elements=2) > 1

    def PeekException(self):
        """Raises an exception if the first iterated element raised."""
        if self._PopulateHead() and self.head[0][0] == 'exception':
            exception_tuple = self.head[0]
            raise six.reraise(exception_tuple[1].__class__, exception_tuple[1], exception_tuple[2])