from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def UseWebSecurityScannerApi(api_version):
    """Mark this command class to use given Web Security Scanner API version.

  Args:
    api_version: Web Security Scanner API version to use for the command

  Returns:
    The decorator function
  """

    def InitApiHolder(cmd_class):
        """Wrapper function for the decorator."""
        cmd_class._wss_version = api_version
        return cmd_class
    return InitApiHolder