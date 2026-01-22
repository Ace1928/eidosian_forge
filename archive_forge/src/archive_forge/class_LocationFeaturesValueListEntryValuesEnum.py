from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationFeaturesValueListEntryValuesEnum(_messages.Enum):
    """LocationFeaturesValueListEntryValuesEnum enum type.

    Values:
      LOCATION_FEATURE_UNSPECIFIED: No publicly supported feature in this
        location
      SITE_TO_CLOUD_SPOKES: Site-to-cloud spokes are supported in this
        location
      SITE_TO_SITE_SPOKES: Site-to-site spokes are supported in this location
    """
    LOCATION_FEATURE_UNSPECIFIED = 0
    SITE_TO_CLOUD_SPOKES = 1
    SITE_TO_SITE_SPOKES = 2