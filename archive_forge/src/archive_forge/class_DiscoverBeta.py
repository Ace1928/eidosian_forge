from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import connection_profiles
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.command_lib.datastream.connection_profiles import flags as cp_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
@base.Deprecate(is_removed=False, warning='Datastream beta version is deprecated. Please use`gcloud datastream connection-profiles discover` command instead.')
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class DiscoverBeta(_Discover, base.Command):
    """Discover a Datastream connection profile."""