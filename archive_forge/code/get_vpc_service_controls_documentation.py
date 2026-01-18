from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import properties
Run 'services vpc-peerings get-vpc-service-controls'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The state of the Vpc Service Controls, that is enabled or disabled.
    