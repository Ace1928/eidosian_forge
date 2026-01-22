from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.policy_intelligence import policy_analyzer
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class QueryActivityAlpha(base.Command):
    """Query activities on cloud resource."""
    detailed_help = _DETAILED_HELP_ALPHA

    @staticmethod
    def Args(parser):
        """Parses arguments for the commands."""
        _ArgsAlpha(parser)

    def Run(self, args):
        return _Run(args)