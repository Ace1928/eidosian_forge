from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Recommendations(base.Group):
    """Commands related to recommendations for resources in terraform.
  """
    category = base.DECLARATIVE_CONFIGURATION_CATEGORY