from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetOracleConnectionProfile(self, args):
    """Creates an Oracle connection profile according to the given args.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      OracleConnectionProfile, to use when creating the connection profile.
    """
    ssl_config = self._GetSslServerOnlyConfig(args)
    connection_profile_obj = self.messages.OracleConnectionProfile(host=args.host, port=args.port, username=args.username, password=args.password, ssl=ssl_config, databaseService=args.database_service)
    private_connectivity_ref = args.CONCEPTS.private_connection.Parse()
    if private_connectivity_ref:
        connection_profile_obj.privateConnectivity = self.messages.PrivateConnectivity(privateConnection=private_connectivity_ref.RelativeName())
    elif args.forward_ssh_hostname:
        connection_profile_obj.forwardSshConnectivity = self._GetForwardSshTunnelConnectivity(args)
    elif args.static_ip_connectivity:
        connection_profile_obj.staticServiceIpConnectivity = {}
    return connection_profile_obj