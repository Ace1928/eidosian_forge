from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutingModeValueValuesEnum(_messages.Enum):
    """Optional. Routing Mode. Default value is set to GLOBAL. For type =
    PRIVATE_SERVICE_ACCESS, this field can be set to GLOBAL or REGIONAL, for
    other types only GLOBAL is supported.

    Values:
      ROUTING_MODE_UNSPECIFIED: The default value. This value should never be
        used.
      GLOBAL: Global Routing Mode
      REGIONAL: Regional Routing Mode
    """
    ROUTING_MODE_UNSPECIFIED = 0
    GLOBAL = 1
    REGIONAL = 2