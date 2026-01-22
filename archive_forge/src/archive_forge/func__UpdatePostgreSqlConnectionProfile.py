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
def _UpdatePostgreSqlConnectionProfile(self, connection_profile, args, update_fields):
    """Updates PostgreSQL connection profile."""
    if args.IsSpecified('host'):
        connection_profile.postgresql.host = args.host
        update_fields.append('postgresql.host')
    if args.IsSpecified('port'):
        connection_profile.postgresql.port = args.port
        update_fields.append('postgresql.port')
    if args.IsSpecified('username'):
        connection_profile.postgresql.username = args.username
        update_fields.append('postgresql.username')
    if args.IsSpecified('password'):
        connection_profile.postgresql.password = args.password
        update_fields.append('postgresql.password')
    if args.IsSpecified(self._InstanceArgName()):
        connection_profile.postgresql.cloudSqlId = args.GetValue(self._InstanceArgName())
        update_fields.append('postgresql.instance')
    if self._api_version == 'v1' and args.IsSpecified('alloydb_cluster'):
        connection_profile.postgresql.alloydbClusterId = args.alloydb_cluster
        update_fields.append('postgresql.alloydb_cluster')
    self._UpdatePostgreSqlSslConfig(connection_profile, args, update_fields)