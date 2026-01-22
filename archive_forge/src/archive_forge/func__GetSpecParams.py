from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
def _GetSpecParams(params):
    """Recursively yields all the params in the spec.

  Args:
    params: List of Argument or ArgumentGroup objects.

  Yields:
    All the Argument objects in the command spec.
  """
    for param in params:
        if isinstance(param, yaml_arg_schema.ArgumentGroup):
            for p in _GetSpecParams(param.arguments):
                yield p
        else:
            yield param