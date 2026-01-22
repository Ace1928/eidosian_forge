from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class SqlIntegrations(base.Group):
    """Discover Cloud SQL integrations with Managed Microsoft AD domains."""