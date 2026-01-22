from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iap import util as iap_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class GetIamPolicyALPHA(GetIamPolicy):
    """Get IAM policy for an IAP IAM resource.

  *{command}* displays the IAM policy associated with an IAP IAM
  resource. If formatted as JSON, the output can be edited and used as a policy
  file for set-iam-policy. The output includes an "etag" field
  identifying the version emitted and allowing detection of
  concurrent policy updates; see
  $ {parent_command} set-iam-policy for additional details.
  """

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        iap_util.AddIapIamResourceArgs(parser, use_region_arg=True)
        base.URI_FLAG.RemoveFromParser(parser)