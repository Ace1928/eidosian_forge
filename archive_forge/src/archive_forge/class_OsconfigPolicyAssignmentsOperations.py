from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class OsconfigPolicyAssignmentsOperations(base.Group):
    """Manage long-running operations generated from creating or editing OS policy assignments."""