from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _check_element(name, element, characters, min_len, max_len):
    """Checks a given named element matches character and length restrictions.

  Args:
    name: the name of the element being validated
    element: the actual element being checked
    characters: acceptable characters for this element, or None
    min_len: minimum element length, or None
    max_len: maximum element length, or None

  Raises:
    BadNameException: one of the restrictions was not met.
  """
    length = len(element)
    if min_len and length < min_len:
        raise BadNameException('Invalid %s: %s, must be at least %s characters' % (name, element, min_len))
    if max_len and length > max_len:
        raise BadNameException('Invalid %s: %s, must be at most %s characters' % (name, element, max_len))
    if element.strip(characters):
        raise BadNameException('Invalid %s: %s, acceptable characters include: %s' % (name, element, characters))