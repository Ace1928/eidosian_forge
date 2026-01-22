from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.certificate_manager import certificates
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.certificate_manager import util
def _TransformSANs(sans, undefined=''):
    """Joins list of SANs with \\n as separator..

  Args:
    sans: list of SANs.
    undefined: str, value to be returned if no SANs are found.

  Returns:
    String representation to be shown in table view.
  """
    return '\n'.join(sans) if sans else undefined