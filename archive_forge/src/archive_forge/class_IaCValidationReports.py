from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class IaCValidationReports(base.Group):
    """Manage Cloud SCC (Security Command Center) iac-validation-reports."""