from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import log as secrets_log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class DestroyBeta(Destroy):
    """Destroy a secret version's metadata and secret data.

  Destroy a secret version's metadata and secret data. This action is
  irreversible.

  ## EXAMPLES

  Destroy version '123' of the secret named 'my-secret':

    $ {command} 123 --secret=my-secret

  Destroy version '123' of the secret named 'my-secret' using an etag:

    $ {command} 123 --secret=my-secret --etag=\\"123\\"
  """

    @staticmethod
    def Args(parser):
        secrets_args.AddVersion(parser, purpose='to destroy', positional=True, required=True)
        secrets_args.AddLocation(parser, purpose='to destroy ', hidden=True)
        secrets_args.AddVersionEtag(parser)

    def Run(self, args):
        api_version = secrets_api.GetApiFromTrack(self.ReleaseTrack())
        version_ref = args.CONCEPTS.version.Parse()
        console_io.PromptContinue(self.CONFIRM_DESTROY_MESSAGE.format(version=version_ref.Name(), secret=version_ref.Parent().Name()), throw_if_unattended=True, cancel_on_no=True)
        result = secrets_api.Versions(api_version=api_version).Destroy(version_ref, etag=args.etag, secret_location=args.location)
        secrets_log.Versions().Destroyed(version_ref)
        return result