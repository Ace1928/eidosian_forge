from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iap import util as iap_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetIamPolicyALPHA(SetIamPolicy):
    """Set the IAM policy for an IAP IAM resource.

  This command replaces the existing IAM policy for an IAP IAM resource, given
  a file encoded in JSON or YAML that contains the IAM policy. If the given
  policy file specifies an "etag" value, then the replacement will succeed only
  if the policy already in place matches that etag. (An etag obtained via
  $ {parent_command} get-iam-policy will prevent the replacement if
  the policy for the resource has been subsequently updated.) A policy
  file that does not contain an etag value will replace any existing policy for
  the resource.
  """

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        iap_util.AddIapIamResourceArgs(parser, use_region_arg=True)
        iap_util.AddIAMPolicyFileArg(parser)
        base.URI_FLAG.RemoveFromParser(parser)