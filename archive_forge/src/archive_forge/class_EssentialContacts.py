from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class EssentialContacts(base.Group):
    """Manage Essential Contacts.

  Essential Contacts can be set on a Cloud resource to receive communications
  from Google Cloud regarding that resource.
  """
    category = base.MANAGEMENT_TOOLS_CATEGORY