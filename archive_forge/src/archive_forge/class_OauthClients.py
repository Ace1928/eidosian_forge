from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class OauthClients(base.Group):
    """Create and manage OAuth clients.

  The {command} group lets you create and manage OAuth clients for
  projects on the Google Cloud Platform.
  """