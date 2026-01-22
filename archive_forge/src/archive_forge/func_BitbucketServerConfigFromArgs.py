from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def BitbucketServerConfigFromArgs(args, update=False):
    """Construct the BitbucketServer resource from the command line args.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    update: bool, if the args are for an update.

  Returns:
    A populated BitbucketServerConfig message.
  """
    messages = GetMessagesModule()
    bbs = messages.BitbucketServerConfig()
    bbs.hostUri = args.host_uri
    bbs.username = args.user_name
    bbs.apiKey = args.api_key
    secret_location = messages.BitbucketServerSecrets()
    secret_location.adminAccessTokenVersionName = args.admin_access_token_secret_version
    secret_location.readAccessTokenVersionName = args.read_access_token_secret_version
    secret_location.webhookSecretVersionName = args.webhook_secret_secret_version
    if update or secret_location is not None:
        bbs.secrets = secret_location
    if not update:
        if args.peered_network is None and args.peered_network_ip_range is not None:
            raise c_exceptions.RequiredArgumentException('peered-network-ip-range', '--peered-network is required when specifying --peered-network-ip-range.')
        if args.peered_network is not None:
            bbs.peeredNetwork = args.peered_network
            bbs.peeredNetworkIpRange = args.peered_network_ip_range
    if args.IsSpecified('ssl_ca_file'):
        bbs.sslCa = args.ssl_ca_file
    return bbs