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
def _UpdateMysqlSslConfig(self, connection_profile, args, update_fields):
    """Updates Mysql SSL config."""
    if args.IsSpecified('client_key'):
        connection_profile.mysqlProfile.sslConfig.clientKey = args.client_key
        update_fields.append('mysqlProfile.sslConfig.clientKey')
    if args.IsSpecified('client_certificate'):
        connection_profile.mysqlProfile.sslConfig.clientCertificate = args.client_certificate
        update_fields.append('mysqlProfile.sslConfig.clientCertificate')
    if args.IsSpecified('ca_certificate'):
        connection_profile.mysqlProfile.sslConfig.caCertificate = args.ca_certificate
        update_fields.append('mysqlProfile.sslConfig.caCertificate')