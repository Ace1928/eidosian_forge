from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataRefreshTypeValueValuesEnum(_messages.Enum):
    """Specifies whether the data source supports automatic data refresh for
    the past few days, and how it's supported. For some data sources, data
    might not be complete until a few days later, so it's useful to refresh
    data automatically.

    Values:
      DATA_REFRESH_TYPE_UNSPECIFIED: The data source won't support data auto
        refresh, which is default value.
      SLIDING_WINDOW: The data source supports data auto refresh, and runs
        will be scheduled for the past few days. Does not allow custom values
        to be set for each transfer config.
      CUSTOM_SLIDING_WINDOW: The data source supports data auto refresh, and
        runs will be scheduled for the past few days. Allows custom values to
        be set for each transfer config.
    """
    DATA_REFRESH_TYPE_UNSPECIFIED = 0
    SLIDING_WINDOW = 1
    CUSTOM_SLIDING_WINDOW = 2