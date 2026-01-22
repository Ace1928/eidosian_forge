from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ConnectorEnforcementValueValuesEnum(_messages.Enum):
    """Specifies if connections must use Cloud SQL connectors. Option values
    include the following: `NOT_REQUIRED` (Cloud SQL instances can be
    connected without Cloud SQL Connectors) and `REQUIRED` (Only allow
    connections that use Cloud SQL Connectors) Note that using REQUIRED
    disables all existing authorized networks. If this field is not specified
    when creating a new instance, NOT_REQUIRED is used. If this field is not
    specified when patching or updating an existing instance, it is left
    unchanged in the instance.

    Values:
      CONNECTOR_ENFORCEMENT_UNSPECIFIED: The requirement for Cloud SQL
        connectors is unknown.
      NOT_REQUIRED: Do not require Cloud SQL connectors.
      REQUIRED: Require all connections to use Cloud SQL connectors, including
        the Cloud SQL Auth Proxy and Cloud SQL Java, Python, and Go
        connectors. Note: This disables all existing authorized networks.
    """
    CONNECTOR_ENFORCEMENT_UNSPECIFIED = 0
    NOT_REQUIRED = 1
    REQUIRED = 2