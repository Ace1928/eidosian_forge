from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultStatusTypeValueValuesEnum(_messages.Enum):
    """Transformation result status type, this will be either SUCCESS, or it
    will be the reason for why the transformation was not completely
    successful.

    Values:
      STATE_TYPE_UNSPECIFIED: Unused.
      INVALID_TRANSFORM: This will be set when a finding could not be
        transformed (i.e. outside user set bucket range).
      BIGQUERY_MAX_ROW_SIZE_EXCEEDED: This will be set when a BigQuery
        transformation was successful but could not be stored back in BigQuery
        because the transformed row exceeds BigQuery's max row size.
      METADATA_UNRETRIEVABLE: This will be set when there is a finding in the
        custom metadata of a file, but at the write time of the transformed
        file, this key / value pair is unretrievable.
      SUCCESS: This will be set when the transformation and storing of it is
        successful.
    """
    STATE_TYPE_UNSPECIFIED = 0
    INVALID_TRANSFORM = 1
    BIGQUERY_MAX_ROW_SIZE_EXCEEDED = 2
    METADATA_UNRETRIEVABLE = 3
    SUCCESS = 4