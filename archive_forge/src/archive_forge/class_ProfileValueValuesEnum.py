from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProfileValueValuesEnum(_messages.Enum):
    """Profile specifies the installation profile for the Anthos bare metal
    cluster.

    Values:
      DEFAULT: Default is the default installation profile.
      EDGE: Edge profile is tailored for edge deployment.
    """
    DEFAULT = 0
    EDGE = 1