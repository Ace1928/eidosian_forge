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
def _GetPostgreSqlConnectionProfile(self, args):
    """Creates a Postgresql connection profile according to the given args.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      PostgreSqlConnectionProfile, to use when creating the connection profile.
    """
    ssl_config = self._GetSslConfig(args)
    alloydb_cluster = args.alloydb_cluster if self._api_version == 'v1' else ''
    connection_profile_obj = self.messages.PostgreSqlConnectionProfile(host=args.host, port=args.port, username=args.username, password=args.password, ssl=ssl_config, cloudSqlId=args.GetValue(self._InstanceArgName()), alloydbClusterId=alloydb_cluster)
    private_service_connect_connectivity_ref = args.CONCEPTS.psc_service_attachment.Parse()
    if private_service_connect_connectivity_ref:
        psc_relative_name = private_service_connect_connectivity_ref.RelativeName()
        connection_profile_obj.privateServiceConnectConnectivity = self.messages.PrivateServiceConnectConnectivity(serviceAttachment=psc_relative_name)
    elif args.static_ip_connectivity:
        connection_profile_obj.staticIpConnectivity = {}
    return connection_profile_obj