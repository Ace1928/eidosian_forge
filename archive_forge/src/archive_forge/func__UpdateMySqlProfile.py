from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _UpdateMySqlProfile(self, connection_profile, args, update_fields):
    """Updates MySQL connection profile."""
    if args.IsSpecified('mysql_hostname'):
        connection_profile.mysqlProfile.hostname = args.mysql_hostname
        update_fields.append('mysqlProfile.hostname')
    if args.IsSpecified('mysql_port'):
        connection_profile.mysqlProfile.port = args.mysql_port
        update_fields.append('mysqlProfile.port')
    if args.IsSpecified('mysql_username'):
        connection_profile.mysqlProfile.username = args.mysql_username
        update_fields.append('mysqlProfile.username')
    if args.IsSpecified('mysql_password'):
        connection_profile.mysqlProfile.password = args.mysql_password
        update_fields.append('mysqlProfile.password')
    self._UpdateMysqlSslConfig(connection_profile, args, update_fields)