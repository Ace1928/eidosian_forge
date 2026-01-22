from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.kms_configs import client as kmsconfigs_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp.kms_configs import flags as kmsconfigs_flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class EncryptBeta(Encrypt):
    """Encrypt all existing volumes and storage pools in the same region with the desired Cloud NetApp Volumes KMS Config."""
    _RELEASE_TRACK = base.ReleaseTrack.BETA