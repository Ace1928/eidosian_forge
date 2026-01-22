from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
@base.Deprecate(is_removed=False, warning=_DEPRECATION_WARNING)
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class SslCerts(base.Group):
    """Provide commands for managing SSL certificates of Cloud SQL instances.

  Provide commands for managing SSL certificates of Cloud SQL instances,
  including creating, deleting, listing, and getting information about
  certificates.
  """
    category = base.DATABASES_CATEGORY