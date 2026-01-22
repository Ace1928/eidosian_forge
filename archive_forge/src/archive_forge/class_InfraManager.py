from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class InfraManager(base.Group):
    """Manage Infra Manager resources."""
    category = base.MANAGEMENT_TOOLS_CATEGORY