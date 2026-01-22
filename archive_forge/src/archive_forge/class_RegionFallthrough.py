from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
class RegionFallthrough(deps.PropertyFallthrough):
    """Custom fallthrough for region dependent on GCF generation.

  For GCF gen1 this falls back to the functions/region property.

  For GCF gen2 the property fallback is only used if it is explicitly set.
  Otherwise the region is prompted for.
  """

    def __init__(self, release_track=base.ReleaseTrack.ALPHA):
        super(RegionFallthrough, self).__init__(properties.VALUES.functions.region)
        self.release_track = release_track

    def _Call(self, parsed_args):
        use_gen1 = not ShouldUseGen2()
        if use_gen1 or self.property.IsExplicitlySet():
            return super(RegionFallthrough, self)._Call(parsed_args)
        if not console_io.CanPrompt():
            raise exceptions.RequiredArgumentException('region', 'You must specify a region. Either use the flag `--region` or set the functions/region property.')
        client = client_v2.FunctionsClient(self.release_track)
        regions = [l.locationId for l in client.ListRegions()]
        idx = console_io.PromptChoice(regions, message='Please specify a region:\n')
        region = regions[idx]
        log.status.Print('To make this the default region, run `gcloud config set functions/region {}`.\n'.format(region))
        return region