from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.util import files
def HandleEscapingInNamePath(deployment_ref, unused_args, request):
    """Sets the request path in the name attribute for various add on commands.

  Replaces the '/' within the deployment name by '%2F' in the install,
  uninstall,
  delete, replace, describe commands.

  Args:
    deployment_ref: reference to the deployment object
    unused_args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del unused_args
    request.name = '{}/deployments/{}'.format(deployment_ref.Parent().RelativeName(), deployment_ref.deploymentsId.replace('/', '%2F'))
    return request