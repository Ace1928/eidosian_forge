from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
class _TransformCall(object):
    """A key transform function call with actual args.

  Attributes:
    name: The transform function name.
    func: The transform function.
    active: The parent projection active level. A transform is active if
      transform.active is None or equal to the caller active level.
    map_transform: If r is a list then apply the transform to each list item
      up to map_transform times. map_transform>1 handles nested lists.
    args: List of function call actual arg strings.
    kwargs: List of function call actual keyword arg strings.
  """

    def __init__(self, name, func, active=0, map_transform=0, args=None, kwargs=None):
        self.name = name
        self.func = func
        self.active = active
        self.map_transform = map_transform
        self.args = args or []
        self.kwargs = kwargs or {}

    def __str__(self):
        args = ['<projecton>' if isinstance(arg, resource_projection_spec.ProjectionSpec) else arg for arg in self.args]
        if self.map_transform > 1:
            prefix = 'map({0}).'.format(self.map_transform)
        elif self.map_transform == 1:
            prefix = 'map().'
        else:
            prefix = ''
        return '{0}{1}({2})'.format(prefix, self.name, ','.join(args))

    def __deepcopy__(self, memo):
        return copy.copy(self)