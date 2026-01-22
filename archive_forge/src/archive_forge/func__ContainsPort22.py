from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.compute import base_classes as compute_base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _ContainsPort22(allowed_ports):
    """Checks if the given list of allowed ports contains port 22.

  Args:
    allowed_ports:

  Returns:

  Raises:
    ValueError:Port value must be of type string.
  """
    for port in allowed_ports:
        try:
            if not isinstance(port, str):
                raise ValueError('Port value must be of type string')
        except ValueError as e:
            print(e)
        if port == '22':
            return True
        if '-' in port:
            start = int(port.split('-')[0])
            end = int(port.split('-')[1])
            if start <= 22 <= end:
                return True
    return False