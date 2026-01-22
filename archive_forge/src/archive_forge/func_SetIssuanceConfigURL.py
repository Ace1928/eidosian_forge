from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def SetIssuanceConfigURL(ref, args, request):
    """Converts the issuance-config argument into a relative URL with project name and location.

  Args:
    ref: Reference to the membership object.
    args: Command line arguments.
    request: API request to be issued.

  Returns:
    Modified request
  """
    del ref
    if not args:
        return request
    if hasattr(args, 'issuance_config') and args.issuance_config and (not args.issuance_config.startswith('projects/')):
        request.certificate.managed.issuanceConfig = ISSUANCE_CONFIG_TEMPLATE.format(request.parent, args.issuance_config)
    return request