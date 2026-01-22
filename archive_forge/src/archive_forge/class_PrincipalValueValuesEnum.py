from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrincipalValueValuesEnum(_messages.Enum):
    """Prinicipal/Identity for whom the role need to assigned.

    Values:
      PRINCIPAL_UNSPECIFIED: Value type is not specified.
      CONNECTOR_SA: Service Account used for Connector workload identity This
        is either the default service account if unspecified or Service
        Account provided by Customers through BYOSA.
    """
    PRINCIPAL_UNSPECIFIED = 0
    CONNECTOR_SA = 1