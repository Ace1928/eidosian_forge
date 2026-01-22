from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DeployPolicies(base.Group):
    """Create and manage Deploy Policy resources for Google Cloud Deploy."""
    category = base.CI_CD_CATEGORY