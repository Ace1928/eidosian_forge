from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import util as iap_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RemoveIamPolicyBindingALPHA(RemoveIamPolicyBinding):
    """Remove IAM policy binding from an IAP IAM resource.

  Removes a policy binding from the IAM policy of an IAP IAM resource. One
  binding consists of a member, a role and an optional condition.
  See $ {parent_command} get-iam-policy for examples of how to
  specify an IAP IAM resource.
  """

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        iap_util.AddIapIamResourceArgs(parser, use_region_arg=True)
        iap_util.AddRemoveIamPolicyBindingArgs(parser)
        base.URI_FLAG.RemoveFromParser(parser)