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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StartIapTunnel(base.Command):
    """Starts an IAP TCP forwarding tunnel."""
    fetch_instance_after_connect_error = False
    support_security_gateway = False

    @classmethod
    def Args(cls, parser):
        iap_tunnel.AddProxyServerHelperArgs(parser)
        flags.INSTANCE_ARG.AddArgument(parser)
        parser.add_argument('instance_port', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=65535), help="The name or number of the instance's port to connect to.")
        local_host_port_help_text = '`LOCAL_HOST:LOCAL_PORT` on which gcloud should bind and listen for connections\nthat should be tunneled.\n\n`LOCAL_PORT` may be omitted, in which case it is treated as 0 and an arbitrary\nunused local port is chosen. The colon also may be omitted in that case.\n\nIf `LOCAL_PORT` is 0, an arbitrary unused local port is chosen.'
        parser.add_argument('--local-host-port', type=lambda arg: arg_parsers.HostPort.Parse(arg, ipv6_enabled=True), default='localhost:0', help=local_host_port_help_text)
        parser.add_argument('--listen-on-stdin', action='store_true', hidden=True, help='Whether to get/put local data on stdin/stdout instead of listening on a socket.  It is an error to specify --local-host-port with this, because that flag has no meaning with this.')
        parser.add_argument('--iap-tunnel-disable-connection-check', default=False, action='store_true', help='Disables the immediate check of the connection.')
        iap_tunnel.AddHostBasedTunnelArgs(parser, cls.support_security_gateway)

    def Run(self, args):
        if args.listen_on_stdin and args.IsSpecified('local_host_port'):
            raise calliope_exceptions.ConflictingArgumentsException('--listen-on-stdin', '--local-host-port')
        target = self._GetTargetArgs(args)
        iap_tunnel_helper = self._CreateIapTunnelHelper(args, target)
        self._CheckNumpyInstalled()
        try:
            iap_tunnel_helper.Run()
        except iap_tunnel_websocket.ConnectionCreationError as e:
            if self._ShouldFetchInstanceAfterConnectError(args.zone) and (not target.host):
                self._FetchInstance(args)
            raise e

    def _ShouldFetchInstanceAfterConnectError(self, zone):
        return self.fetch_instance_after_connect_error and zone

    def _CreateIapTunnelHelper(self, args, target):
        if self.support_security_gateway and args.security_gateway:
            tunneler = iap_tunnel.SecurityGatewayTunnelHelper(args, project=target.project, region=target.region, security_gateway=target.security_gateway, host=target.host, port=target.port)
        elif target.host:
            tunneler = iap_tunnel.IAPWebsocketTunnelHelper(args, target.project, region=target.region, network=target.network, host=target.host, port=target.port, dest_group=target.dest_group)
        else:
            tunneler = iap_tunnel.IAPWebsocketTunnelHelper(args, target.project, zone=target.zone, instance=target.instance, interface=target.interface, port=target.port)
        if args.listen_on_stdin:
            iap_tunnel_helper = iap_tunnel.IapTunnelStdinHelper(tunneler)
        else:
            local_host, local_port = self._GetLocalHostPort(args)
            check_connection = True
            if hasattr(args, 'iap_tunnel_disable_connection_check'):
                check_connection = not args.iap_tunnel_disable_connection_check
            iap_tunnel_helper = iap_tunnel.IapTunnelProxyServerHelper(local_host, local_port, check_connection, tunneler)
        return iap_tunnel_helper

    def _GetTargetArgs(self, args):
        if args.IsSpecified('network') and args.IsSpecified('region'):
            return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), region=args.region, network=args.network, host=args.instance_name, port=args.instance_port, dest_group=args.dest_group, zone=None, instance=None, interface=None, security_gateway=None)
        if self.support_security_gateway and args.security_gateway:
            return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), host=args.instance_name, port=args.instance_port, region=args.region, security_gateway=args.security_gateway, network=None, dest_group=None, zone=None, instance=None, interface=None)
        if self._ShouldFetchInstanceAfterConnectError(args.zone):
            return _CreateTargetArgs(project=properties.VALUES.core.project.GetOrFail(), zone=args.zone, instance=args.instance_name, interface='nic0', port=args.instance_port, region=None, network=None, host=None, dest_group=None, security_gateway=None)
        instance_ref, instance_obj = self._FetchInstance(args)
        return _CreateTargetArgs(project=instance_ref.project, zone=instance_ref.zone, instance=instance_obj.name, interface=ssh_utils.GetInternalInterface(instance_obj).name, port=args.instance_port, region=None, network=None, host=None, dest_group=None, security_gateway=None)

    def _FetchInstance(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        ssh_helper = ssh_utils.BaseSSHCLIHelper()
        instance_ref = flags.SSH_INSTANCE_RESOLVER.ResolveResources([args.instance_name], scope.ScopeEnum.ZONE, args.zone, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))[0]
        return (instance_ref, ssh_helper.GetInstance(client, instance_ref))

    def _GetLocalHostPort(self, args):
        local_host_arg = args.local_host_port.host or 'localhost'
        port_arg = int(args.local_host_port.port) if args.local_host_port.port else 0
        local_port = iap_tunnel.DetermineLocalPort(port_arg=port_arg)
        if not port_arg:
            log.status.Print('Picking local unused port [%d].' % local_port)
        return (local_host_arg, local_port)

    def _CheckNumpyInstalled(self):
        try:
            import numpy
        except ImportError:
            log.warning(_NUMPY_HELP_TEXT)