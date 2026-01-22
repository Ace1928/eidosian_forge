from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.sole_tenancy.node_groups import flags
@staticmethod
def _Flags(parser):
    """Adds the flags for this command.

    Removes the URI flag since nodes don't have URIs.

    Args:
      parser: The argparse parser.
    """
    base.ListCommand._Flags(parser)
    base.URI_FLAG.RemoveFromParser(parser)