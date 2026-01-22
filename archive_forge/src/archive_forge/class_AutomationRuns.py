from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AutomationRuns(base.Group):
    """Manages AutomationRuns resources for Cloud Deploy."""
    category = base.CI_CD_CATEGORY