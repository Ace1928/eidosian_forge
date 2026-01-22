from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.asset import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AnalyzeIamPolicyGA(base.Command):
    """Analyzes IAM policies that match a request."""
    detailed_help = DETAILED_HELP
    _API_VERSION = client_util.DEFAULT_API_VERSION

    @classmethod
    def Args(cls, parser):
        flags.AddAnalyzerParentArgs(parser)
        flags.AddAnalyzerSelectorsGroup(parser)
        flags.AddAnalyzerOptionsGroup(parser, True)
        flags.AddAnalyzerConditionContextGroup(parser)
        flags.AddAnalyzerSavedAnalysisQueryArgs(parser)

    def Run(self, args):
        client = client_util.AnalyzeIamPolicyClient(self._API_VERSION)
        return client.Analyze(args)