from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
def _ApplyTransformation(components, func, obj):
    """Applies the given function to the property pointed to by components.

  For example:

      obj = {'x': {'y': 1, 'z': 2}, 'y': [1, 2, 3]}
      _ApplyTransformation(_Parse('x.y'), lambda x: x* 2, obj)

  results in obj becoming:

      {'x': {'y': 2, 'z': 2}, 'y': [1, 2, 3]}

  Args:
    components: A parsed property.
    func: The function to apply.
    obj: A JSON-serializable dict to apply the function to.
  """
    if isinstance(obj, dict) and isinstance(components[0], _Key):
        val = obj.get(components[0])
        if val is None:
            return
        if len(components) == 1:
            obj[components[0]] = func(val)
        else:
            _ApplyTransformation(components[1:], func, val)
    elif isinstance(obj, list) and isinstance(components[0], _Index):
        idx = components[0]
        if idx > len(obj) - 1:
            return
        if len(components) == 1:
            obj[idx] = func(obj[idx])
        else:
            _ApplyTransformation(components[1:], func, obj[idx])
    elif isinstance(obj, list) and isinstance(components[0], _Slice):
        for i, val in enumerate(obj):
            if len(components) == 1:
                obj[i] = func(val)
            else:
                _ApplyTransformation(components[1:], func, val)