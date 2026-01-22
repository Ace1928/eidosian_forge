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
def GitHubEnterpriseConfigFromArgs(args, update=False):
    """Construct the GitHubEnterpriseConfig resource from the command line args.

  Args:
    args: An argparse namespace. All the arguments that were provided to this
        command invocation.
      update: bool, if the args are for an update.

  Returns:
    A populated GitHubEnterpriseConfig message.
  """
    messages = GetMessagesModule()
    ghe = messages.GitHubEnterpriseConfig()
    ghe.hostUrl = args.host_uri
    ghe.appId = args.app_id
    if args.webhook_key is not None:
        ghe.webhookKey = args.webhook_key
    if not update and args.peered_network is not None:
        ghe.peeredNetwork = args.peered_network
    if args.gcs_bucket is not None:
        gcs_location = messages.GCSLocation()
        gcs_location.bucket = args.gcs_bucket
        gcs_location.object = args.gcs_object
        if args.generation is not None:
            gcs_location.generation = args.generation
        ghe.appConfigJson = gcs_location
    else:
        secret_location = messages.GitHubEnterpriseSecrets()
        secret_location.privateKeyName = args.private_key_name
        secret_location.webhookSecretName = args.webhook_secret_name
        secret_location.oauthSecretName = args.oauth_secret_name
        secret_location.oauthClientIdName = args.oauth_client_id_name
        ghe.secrets = secret_location
    return ghe