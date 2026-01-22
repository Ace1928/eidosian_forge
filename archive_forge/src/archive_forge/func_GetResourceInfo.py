from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def GetResourceInfo(request):
    """Returns a tuple of the resource and resource name from the `request`.

  Args:
    request: A Request object instance.

  Returns:
    A tuple of the resource string path and the resource name.

  Raises:
    UnexpectedResourceField: The `request` had a unsupported resource.
  """
    resource = ''
    resource_name = ''
    if hasattr(request, 'parent'):
        resource = request.parent
        resource_name = 'parent'
    elif hasattr(request, 'name'):
        resource = request.name
        resource_name = 'name'
    elif hasattr(request, 'subscription'):
        resource = request.subscription
        resource_name = 'subscription'
    else:
        raise UnexpectedResourceField('The resource specified for this command is unknown!')
    return (resource, resource_name)