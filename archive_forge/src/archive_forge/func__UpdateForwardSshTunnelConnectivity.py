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
def _UpdateForwardSshTunnelConnectivity(self, connection_profile, args, update_fields):
    """Updates Forward SSH tunnel connectivity config."""
    if args.IsSpecified('forward_ssh_hostname'):
        connection_profile.forwardSshConnectivity.hostname = args.forward_ssh_hostname
        update_fields.append('forwardSshConnectivity.hostname')
    if args.IsSpecified('forward_ssh_port'):
        connection_profile.forwardSshConnectivity.port = args.forward_ssh_port
        update_fields.append('forwardSshConnectivity.port')
    if args.IsSpecified('forward_ssh_username'):
        connection_profile.forwardSshConnectivity.username = args.forward_ssh_username
        update_fields.append('forwardSshConnectivity.username')
    if args.IsSpecified('forward_ssh_private_key'):
        connection_profile.forwardSshConnectivity.privateKey = args.forward_ssh_private_key
        update_fields.append('forwardSshConnectivity.privateKey')
    if args.IsSpecified('forward_ssh_password'):
        connection_profile.forwardSshConnectivity.privateKey = args.forward_ssh_password
        update_fields.append('forwardSshConnectivity.password')