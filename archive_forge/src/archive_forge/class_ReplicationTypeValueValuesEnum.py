from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReplicationTypeValueValuesEnum(_messages.Enum):
    """The type of replication this instance uses. This can be either
    `ASYNCHRONOUS` or `SYNCHRONOUS`. (Deprecated) This property was only
    applicable to First Generation instances.

    Values:
      SQL_REPLICATION_TYPE_UNSPECIFIED: This is an unknown replication type
        for a Cloud SQL instance.
      SYNCHRONOUS: The synchronous replication mode for First Generation
        instances. It is the default value.
      ASYNCHRONOUS: The asynchronous replication mode for First Generation
        instances. It provides a slight performance gain, but if an outage
        occurs while this option is set to asynchronous, you can lose up to a
        few seconds of updates to your data.
    """
    SQL_REPLICATION_TYPE_UNSPECIFIED = 0
    SYNCHRONOUS = 1
    ASYNCHRONOUS = 2