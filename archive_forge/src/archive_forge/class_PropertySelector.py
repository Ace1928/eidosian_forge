from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
class PropertySelector(object):
    """Extracts and/or transforms values in JSON-serializable dicts.

  For example:

      selector = PropertySelector(
          properties=['x.y', 'y[0]'],
          transformations=[
              ('x.y', lambda x: x + 5),
              ('y[]', lambda x: x * 5),
      ])
      selector.SelectProperties(
          {'x': {'y': 1, 'z': 2}, 'y': [1, 2, 3]})

  returns:

      collections.OrderedDict([
          ('x', collections.OrderedDict([('y', 6)])),
          ('y', [5])
      ])

  Items are extracted in the order requested. Transformations are applied
  in the order they appear.
  """

    def __init__(self, properties=None, transformations=None):
        """Creates a new PropertySelector with the given properties."""
        if properties:
            self._compiled_properties = [_Parse(p) for p in properties]
        else:
            self._compiled_properties = None
        if transformations:
            self._compiled_transformations = [(_Parse(p), func) for p, func in transformations]
        else:
            self._compiled_transformations = None
        self.properties = properties
        self.transformations = transformations

    def Apply(self, obj):
        """An OrderedDict resulting from filtering and transforming obj."""
        if self._compiled_properties:
            res = _Filter(obj, self._compiled_properties) or collections.OrderedDict()
        else:
            res = _DictToOrderedDict(obj)
        if self._compiled_transformations:
            for compiled_property, func in self._compiled_transformations:
                _ApplyTransformation(compiled_property, func, res)
        return res