from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.data_catalog import search
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class SearchBeta(Search):
    __doc__ = Search.__doc__

    def Run(self, args):
        """Run the search command."""
        version_label = 'v1beta1'
        return search.Search(args, version_label)