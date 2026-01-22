from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
class ConflictingProperties(Error):
    """Raised when a property conflicts with another.

  Examples of conflicting properties:

      - "a.b" and "a[0].b"
      - "a[0].b" and "a[].b"
  """