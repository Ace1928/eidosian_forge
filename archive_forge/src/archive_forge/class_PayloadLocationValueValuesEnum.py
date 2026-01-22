from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PayloadLocationValueValuesEnum(_messages.Enum):
    """Location within the request where the payload was placed.

    Values:
      LOCATION_UNSPECIFIED: Unknown Location.
      COMPLETE_REQUEST_BODY: The XML payload replaced the complete request
        body.
    """
    LOCATION_UNSPECIFIED = 0
    COMPLETE_REQUEST_BODY = 1