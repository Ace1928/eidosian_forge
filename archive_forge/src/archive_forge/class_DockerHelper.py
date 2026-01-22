from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import credential_utils
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DockerHelper(base.Command):
    """A Docker credential helper to provide access to GCR repositories."""
    GET = 'get'
    LIST = 'list'

    @staticmethod
    def Args(parser):
        parser.add_argument('method', help='The docker credential helper method.')
        parser.display_info.AddFormat('json')

    def Run(self, args):
        """Run the helper command."""
        if args.method == DockerHelper.LIST:
            return {'https://' + url: '_dcgcloud_token' for url in credential_utils.DefaultAuthenticatedRegistries()}
        elif args.method == DockerHelper.GET:
            try:
                cred = c_store.Load(use_google_auth=True)
            except creds_exceptions.NoActiveAccountException:
                log.Print('You do not currently have an active account selected. See https://cloud.google.com/sdk/docs/authorizing for more information.')
                sys.exit(1)
            c_store.RefreshIfExpireWithinWindow(cred, window=TOKEN_MIN_LIFETIME)
            url = sys.stdin.read().strip()
            if url.replace('https://', '', 1) not in credential_utils.SupportedRegistries():
                raise exceptions.Error('Repository url [{url}] is not supported'.format(url=url))
            token = cred.token if c_creds.IsGoogleAuthCredentials(cred) else cred.access_token
            return {'Secret': token, 'Username': '_dcgcloud_token'}
        args.GetDisplayInfo().AddFormat('none')
        return None