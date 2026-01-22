from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class KmsConfigsBeta(KmsConfigs):
    """Create and manage Cloud NetApp Volumes KMS Configs."""