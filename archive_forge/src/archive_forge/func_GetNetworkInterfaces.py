from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def GetNetworkInterfaces(self, args, resources, client, holder, project, location, scope, skip_defaults):
    if args.network_interface:
        return create_utils.CreateNetworkInterfaceMessages(resources=resources, compute_client=client, network_interface_arg=args.network_interface, project=project, location=location, scope=scope, support_internal_ipv6_reservation=self._support_internal_ipv6_reservation)
    return self._GetNetworkInterfaces(args, client, holder, project, location, scope, skip_defaults)