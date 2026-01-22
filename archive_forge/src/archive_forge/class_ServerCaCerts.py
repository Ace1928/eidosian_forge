from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ServerCaCerts(base.Group):
    """Provide commands for managing server CA certs of Cloud SQL instances.

  Provide commands for managing server CA certs of Cloud SQL instances,
  including creating, listing, rotating in, and rolling back certs.
  """