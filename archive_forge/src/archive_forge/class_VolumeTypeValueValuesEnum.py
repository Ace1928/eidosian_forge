from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeTypeValueValuesEnum(_messages.Enum):
    """Optional. Type of the EBS volume. When unspecified, it defaults to GP2
    volume.

    Values:
      VOLUME_TYPE_UNSPECIFIED: Not set.
      GP2: GP2 (General Purpose SSD volume type).
      GP3: GP3 (General Purpose SSD volume type).
    """
    VOLUME_TYPE_UNSPECIFIED = 0
    GP2 = 1
    GP3 = 2