from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.asset import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AnalyzeIamPolicyALPHA(AnalyzeIamPolicyBETA):
    """ALPHA version, Analyzes IAM policies that match a request."""

    @classmethod
    def Args(cls, parser):
        AnalyzeIamPolicyBETA.Args(parser)
        options_group = flags.GetOrAddOptionGroup(parser)
        flags.AddAnalyzerIncludeDenyPolicyAnalysisArgs(options_group)