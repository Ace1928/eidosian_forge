from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitHubEnterpriseConfigArgs(parser, update=False):
    """Sets up all the argparse flags for creating or updating a GHE config.

  Args:
    parser: An argparse.ArgumentParser-like object.
    update: If true, use the version of the flags for updating a config.
      Otherwise, use the version for creating a config.

  Returns:
    The parser argument with GitHub Enterprise config flags added in.
  """
    parser.add_argument('--host-uri', required=not update, help='The host uri of the GitHub Enterprise Server.')
    parser.add_argument('--app-id', type=int, required=not update, help='The app id of the GitHub app that should be associated with this config.')
    if not update:
        parser.add_argument('--peered-network', help='VPC network that should be used when making calls to the GitHub Enterprise Server.\n\nIf not specified, calls will be made over the public internet.\n')
    if update:
        parser.add_argument('CONFIG', help='The unique identifier of the GitHub Enterprise Config to be updated.')
    parser.add_argument('--webhook-key', help="The unique identifier that Cloud Build expects to be set as the value for\nthe query field `webhook_key` on incoming webhook requests.\n\nIf this is not set, Cloud Build will generate one on the user's behalf.\n")
    gcs_or_secretmanager = parser.add_mutually_exclusive_group(required=not update)
    gcs = gcs_or_secretmanager.add_argument_group('Cloud Storage location of the GitHub App credentials:')
    gcs.add_argument('--gcs-bucket', required=True, help='The Cloud Storage bucket containing the credential payload.')
    gcs.add_argument('--gcs-object', required=True, help='The Cloud Storage object containing the credential payload.')
    gcs.add_argument('--generation', type=int, help='The object generation to read the credential payload from.\n\nIf this is not set, Cloud Build will read the latest version.\n')
    secretmanager = gcs_or_secretmanager.add_argument_group('Secret Manager resources of the GitHub App credentials:')
    secretmanager.add_argument('--private-key-name', required=True, help='Secret Manager resource containing the private key.')
    secretmanager.add_argument('--webhook-secret-name', required=True, help='Secret Manager resource containing the webhook key.')
    secretmanager.add_argument('--oauth-secret-name', required=True, help='Secret Manager resource containing the oauth secret.')
    secretmanager.add_argument('--oauth-client-id-name', required=True, help='Secret Manager resource containing the oauth client id.')
    secretmanager.add_argument('--private-key-version-name', help='Secret Manager SecretVersion resource containing the private key.')
    secretmanager.add_argument('--webhook-secret-version-name', help='Secret Manager SecretVersion resource containing the webhook key.')
    secretmanager.add_argument('--oauth-secret-version-name', help='Secret Manager SecretVersion resource containing the oauth secret.')
    secretmanager.add_argument('--oauth-client-id-version-name', help='Secret Manager SecretVersion resource containing the oauth client id.')
    return parser