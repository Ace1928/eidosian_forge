from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Datastream(base.Group):
    """Manage Cloud Datastream resources.

  Commands for managing Cloud Datastream resources.
  """
    category = base.DATABASES_CATEGORY

    def Filter(self, context, args):
        del context, args