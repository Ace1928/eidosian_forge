from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BundleOperationalStatusValueValuesEnum(_messages.Enum):
    """The operational status of the bundle interface.

    Values:
      BUNDLE_OPERATIONAL_STATUS_DOWN: If bundleAggregationType is LACP: LACP
        is not established and/or all links in the bundle have DOWN
        operational status. If bundleAggregationType is STATIC: one or more
        links in the bundle has DOWN operational status.
      BUNDLE_OPERATIONAL_STATUS_UP: If bundleAggregationType is LACP: LACP is
        established and at least one link in the bundle has UP operational
        status. If bundleAggregationType is STATIC: all links in the bundle
        (typically just one) have UP operational status.
    """
    BUNDLE_OPERATIONAL_STATUS_DOWN = 0
    BUNDLE_OPERATIONAL_STATUS_UP = 1