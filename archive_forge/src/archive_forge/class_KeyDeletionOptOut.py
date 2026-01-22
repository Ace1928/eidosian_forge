from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class KeyDeletionOptOut(base.Command):
    """Opt out a project of the key deletion change announced via MSA.

  The flag `--project` indicates the project that you want to opt out.

  The optional flag `--undo` allows you to opt a project back in.

  ## EXAMPLES
  The following command opts out the project `my-project-id-or-number`.

    $ {command} --project=projects/my-project-id-or-number

  The following command opts the project `my-project-id-or-number` back in.

    $ {command} \\
        --project=projects/my-project-id-or-number \\
        --undo=true
  """

    @staticmethod
    def Args(parser):
        flags.AddProjectFlag(parser)
        flags.AddUndoOptOutFlag(parser)

    def _CreateSetOptOutRequest(self, args):
        messages = cloudkms_base.GetMessagesAlphaModule()
        req = messages.CloudkmsProjectsSetProjectOptOutStateRequest(name=args.project, setProjectOptOutStateRequest=messages.SetProjectOptOutStateRequest(value=not args.undo))
        return req

    def _VerifyResponseIntegrity(self, req, resp):
        """Verifies that the opt-out preference has been updated correctly.

    Args:
      req: messages.CloudkmsProjectsSetProjectOptOutStateRequest() object
      resp: messages.SetProjectOptOutStateResponse() object.

    Returns:
      Void.
    Raises:
      e2e_integrity.ClientSideIntegrityVerificationError if response integrity
      verification fails.
    """
        if bool(req.setProjectOptOutStateRequest.value) != bool(resp.value):
            raise e2e_integrity.ClientSideIntegrityVerificationError('Your opt-out preference could not be updated correctly. Please try again.')

    def Run(self, args):
        client = cloudkms_base.GetClientAlphaInstance()
        req = self._CreateSetOptOutRequest(args)
        resp = client.projects.SetProjectOptOutState(req)
        self._VerifyResponseIntegrity(req, resp)
        log.WriteToFileOrStdout('-', 'Your opt-out preference has been updated successfully. Opt-out preference: {0}\n'.format(bool(resp.value)))