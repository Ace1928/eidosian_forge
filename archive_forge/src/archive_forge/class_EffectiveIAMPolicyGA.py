from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class EffectiveIAMPolicyGA(base.Command):
    """Get effective IAM policies for a specified list of resources within accessible scope, such as a project, folder or organization."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        AddNamesArgument(parser)
        AddScopeArgument(parser)

    def Run(self, args):
        client = client_util.EffectiveIAMPolicyClient()
        return client.BatchGetEffectiveIAMPolicies(args)