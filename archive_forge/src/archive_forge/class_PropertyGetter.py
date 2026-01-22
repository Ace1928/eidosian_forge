from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
class PropertyGetter(object):
    """Extracts a single field from JSON-serializable dicts.

  For example:

      getter = PropertyGetter('x.y')
      getter.Get({'x': {'y': 1, 'z': 2}, 'y': [1, 2, 3]})

  returns:

      1
  """

    def __init__(self, p):
        """Creates a new PropertyGetter with the given property."""
        self._compiled_property = _Parse(p)

    def Get(self, obj):
        """Returns the property in obj or None if the property does not exist."""
        return copy.deepcopy(_GetProperty(obj, self._compiled_property))