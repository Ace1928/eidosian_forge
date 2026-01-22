from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetTargetArgs(self, args):
    if args.IsSpecified('network') and args.IsSpecified('region'):
        return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), region=args.region, network=args.network, host=args.instance_name, port=args.instance_port, dest_group=args.dest_group, zone=None, instance=None, interface=None, security_gateway=None)
    if self.support_security_gateway and args.security_gateway:
        return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), host=args.instance_name, port=args.instance_port, region=args.region, security_gateway=args.security_gateway, network=None, dest_group=None, zone=None, instance=None, interface=None)
    if self._ShouldFetchInstanceAfterConnectError(args.zone):
        return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), zone=args.zone, instance=args.instance_name, interface='nic0', port=args.instance_port, region=None, network=None, host=None, dest_group=None, security_gateway=None)
    instance_ref, instance_obj = self._FetchInstance(args)
    return _CreateTargetArgs(project=instance_ref.project, zone=instance_ref.zone, instance=instance_obj.name, interface=ssh_utils.GetInternalInterface(instance_obj).name, port=args.instance_port, region=None, network=None, host=None, dest_group=None, security_gateway=None)