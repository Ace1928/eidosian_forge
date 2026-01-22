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
def _UpdateOracleProfile(self, connection_profile, args, update_fields):
    """Updates Oracle connection profile."""
    if args.IsSpecified('oracle_hostname'):
        connection_profile.oracleProfile.hostname = args.oracle_hostname
        update_fields.append('oracleProfile.hostname')
    if args.IsSpecified('oracle_port'):
        connection_profile.oracleProfile.port = args.oracle_port
        update_fields.append('oracleProfile.port')
    if args.IsSpecified('oracle_username'):
        connection_profile.oracleProfile.username = args.oracle_username
        update_fields.append('oracleProfile.username')
    if args.IsSpecified('oracle_password'):
        connection_profile.oracleProfile.password = args.oracle_password
        update_fields.append('oracleProfile.password')
    if args.IsSpecified('database_service'):
        connection_profile.oracleProfile.databaseService = args.database_service
        update_fields.append('oracleProfile.databaseService')