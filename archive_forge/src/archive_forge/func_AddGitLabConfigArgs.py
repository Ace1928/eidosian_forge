from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddGitLabConfigArgs(parser, update=False):
    """Set up all the argparse flags for creating or updating a GitLab config.

  Args:
    parser: An argparse.ArgumentParser-like object.
    update: If true, use the version of the flags for updating a config.
      Otherwise, use the version for creating a config.

  Returns:
    The parser argument with GitLab config flags added in.
  """
    parser.add_argument('--host-uri', required=not update, help='The host uri of the GitLab Enterprise instance.')
    parser.add_argument('--user-name', required=not update, help='The GitLab user name that should be associated with this config.')
    parser.add_argument('--api-key-secret-version', required=not update, help='Secret Manager resource containing the Cloud Build API key that should be associated with this config. The secret is specified in resource URL format projects/{secret_project}/secrets/{secret_name}/versions/{secret_version}.')
    parser.add_argument('--api-access-token-secret-version', required=not update, help='Secret Manager resource containing the API access token. The secret is specified in resource URL format projects/{secret_project}/secrets/{secret_name}/versions/{secret_version}.')
    parser.add_argument('--read-access-token-secret-version', required=not update, help='Secret Manager resource containing the read access token. The secret is specified in resource URL format projects/{secret_project}/secrets/{secret_name}/versions/{secret_version}.')
    parser.add_argument('--webhook-secret-secret-version', required=not update, help='Secret Manager resource containing the webhook secret. The secret is specified in resource URL format projects/{secret_project}/secrets/{secret_name}/versions/{secret_version}.')
    parser.add_argument('--ssl-ca-file', type=arg_parsers.FileContents(), help='Path to a local file that contains SSL certificate to use for requests to GitLab Enterprise. The certificate should be in PEM format.')
    parser.add_argument('--service-directory-service', help='Service Directory service that should be used when making calls to the GitLab Enterprise instance.\n\nIf not specified, calls will be made over the public internet.\n')
    if not update:
        parser.add_argument('--name', required=True, help='The name of the GitLab config.')
        parser.add_argument('--region', required=True, help='The Cloud location of the GitLab config.')
    return parser