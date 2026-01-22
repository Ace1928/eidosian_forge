from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class InvalidateCdnCache(base.SilentCommand):
    """Invalidate specified objects for a URL map in Cloud CDN caches."""
    detailed_help = _DetailedHelp()
    URL_MAP_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser, cust_metavar='URLMAP')
        _Args(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.URL_MAP_ARG)