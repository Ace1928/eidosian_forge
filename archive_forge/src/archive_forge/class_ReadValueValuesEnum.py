from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadValueValuesEnum(_messages.Enum):
    """Output only. Describes the read access mechanism of the data. Not user
    settable.

    Values:
      ACCESS_MODE_UNSPECIFIED: Access mode unspecified.
      DIRECT: Default. Data is accessed directly using storage APIs.
      MANAGED: Data is accessed through a managed interface using BigQuery
        APIs.
    """
    ACCESS_MODE_UNSPECIFIED = 0
    DIRECT = 1
    MANAGED = 2