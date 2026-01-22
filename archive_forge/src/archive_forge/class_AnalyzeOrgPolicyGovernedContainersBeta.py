from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class AnalyzeOrgPolicyGovernedContainersBeta(AnalyzeOrgPolicyGovernedContainers):
    """Analyze organization policies governed containers under a scope."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        AddScopeArgument(parser)
        AddConstraintArgument(parser)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        client = client_util.OrgPolicyAnalyzerClient()
        return client.AnalyzeOrgPolicyGovernedContainers(args)