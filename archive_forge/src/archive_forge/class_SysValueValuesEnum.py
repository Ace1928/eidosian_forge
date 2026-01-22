from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SysValueValuesEnum(_messages.Enum):
    """Trusted attributes supplied by any service that owns resources and
    uses the IAM system for access control.

    Values:
      NO_ATTR: Default non-attribute type
      REGION: Region of the resource
      SERVICE: Service name
      NAME: Resource name
      IP: IP address of the caller
    """
    NO_ATTR = 0
    REGION = 1
    SERVICE = 2
    NAME = 3
    IP = 4