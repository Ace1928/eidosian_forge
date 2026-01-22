from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CloudSecrets(base.Group):
    """Manage locations of users' secrets.

  Manage locations of users' secrets on Google Cloud.
  """

    def Filter(self, context, args):
        del context, args