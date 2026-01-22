from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformZone(r, undefined=''):
    """Returns a zone name from a selfLink.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    A zone name for selfLink from r.
  """
    project = resource_transform.TransformScope(resource_transform.GetKeyValue(r, 'selfLink', ''), 'zones').split('/')[0]
    return project or undefined