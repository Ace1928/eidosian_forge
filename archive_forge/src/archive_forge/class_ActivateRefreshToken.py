from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.auth import refresh_token
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.Hidden
class ActivateRefreshToken(base.SilentCommand):
    """Get credentials via an existing refresh token.

  Use an oauth2 refresh token to manufacture credentials for Google APIs. This
  token must have been acquired via some legitimate means to work. The account
  provided is only used locally to help the Cloud SDK keep track of the new
  credentials, so you can activate, list, and revoke the credentials in the
  future.
  """

    @staticmethod
    def Args(parser):
        """Set args for gcloud auth activate-refresh-token."""
        parser.add_argument('account', help='The account to associate with the refresh token.')
        parser.add_argument('token', nargs='?', help='OAuth2 refresh token. If blank, prompt for value.')

    def Run(self, args):
        """Run the authentication command."""
        token = args.token or console_io.PromptResponse('Refresh token: ')
        if not token:
            raise c_exc.InvalidArgumentException('token', 'No value provided.')
        refresh_token.ActivateCredentials(args.account, token)
        project = args.project
        if project:
            properties.PersistProperty(properties.VALUES.core.project, project)
        log.status.Print('Activated refresh token credentials: [{0}]'.format(args.account))