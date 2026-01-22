from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ExecutionsBeta(base.Group):
    """Manage your Cloud Workflow execution resources."""
    category = base.TOOLS_CATEGORY