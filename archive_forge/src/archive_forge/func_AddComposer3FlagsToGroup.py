from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddComposer3FlagsToGroup(update_type_group):
    """Adds Composer 3 flags to an update group.

  Args:
    update_type_group: argument group, the group to which flags should be added.
  """
    SUPPORT_WEB_SERVER_PLUGINS.AddToParser(update_type_group)
    private_builds_only_group = update_type_group.add_argument_group(mutex=True)
    ENABLE_PRIVATE_BUILDS_ONLY.AddToParser(private_builds_only_group)
    DISABLE_PRIVATE_BUILDS_ONLY.AddToParser(private_builds_only_group)
    vpc_connectivity_group = update_type_group.add_argument_group(mutex=True)
    NETWORK_ATTACHMENT.AddToParser(vpc_connectivity_group)
    DISABLE_VPC_CONNECTIVITY.AddToParser(vpc_connectivity_group)
    network_subnetwork_group = vpc_connectivity_group.add_group(help='Virtual Private Cloud networking')
    NETWORK_FLAG.AddToParser(network_subnetwork_group)
    SUBNETWORK_FLAG.AddToParser(network_subnetwork_group)
    ENABLE_PRIVATE_ENVIRONMENT_UPDATE_FLAG.AddToParser(update_type_group)
    DISABLE_PRIVATE_ENVIRONMENT_UPDATE_FLAG.AddToParser(update_type_group)