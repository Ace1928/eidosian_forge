from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AuditManager(base.Group):
    """Enroll resources, audit workloads and generate reports."""
    category = base.SECURITY_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()