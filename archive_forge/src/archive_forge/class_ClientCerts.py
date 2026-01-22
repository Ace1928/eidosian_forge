from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ClientCerts(base.Group):
    """Provide commands for managing client certificates of Cloud SQL instances.

  Provide commands for managing client certificates of Cloud SQL instances,
  including creating, deleting, listing, and getting information about
  certificates.
  """