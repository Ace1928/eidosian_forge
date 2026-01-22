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
def _ParseTransform(self, func_name, active=0, map_transform=None):
    """Parses a transform function call.

    The initial '(' has already been consumed by the caller.

    Args:
      func_name: The transform function name.
      active: The transform active level or None if always active.
      map_transform: Apply the transform to each resource list item this many
        times.

    Returns:
      A _TransformCall object. The caller appends these to a list that is used
      to apply the transform functions.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.
    """
    here = self.GetPosition()
    func = self._defaults.symbols.get(func_name)
    if not func:
        raise resource_exceptions.UnknownTransformError('Unknown transform function {0} [{1}].'.format(func_name, self.Annotate(here)))
    args = []
    kwargs = {}
    doc = getattr(func, '__doc__', None)
    if doc and resource_projection_spec.PROJECTION_ARG_DOC in doc:
        args.append(self._defaults)
    if getattr(func, '__defaults__', None):
        for arg in self.Args():
            name, sep, val = arg.partition('=')
            if sep:
                kwargs[name] = val
            else:
                args.append(arg)
    else:
        args += self.Args()
    return _TransformCall(func_name, func, active=active, map_transform=map_transform, args=args, kwargs=kwargs)