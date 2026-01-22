from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Vulnerabilities(base.Group):
    """Manage Artifact Vulnerabilities.
  """
    category = base.CI_CD_CATEGORY