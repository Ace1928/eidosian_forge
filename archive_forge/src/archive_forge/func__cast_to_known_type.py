from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _cast_to_known_type(name):
    """Canonicalizes a string representing a type if possible.

  # TODO(dbieber): Support additional canonicalization, such as string/str, and
  # boolean/bool.

  Example:
    _cast_to_known_type("str.") == "str"

  Args:
    name: A string representing a type, or None.
  Returns:
    A canonicalized version of the type string.
  """
    if name is None:
        return None
    return name.rstrip('.')