from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrincipalStateValueValuesEnum(_messages.Enum):
    """Output only. Principal evaluation states indicating whether the
    principals match.

    Values:
      PRINCIPAL_STATE_UNSPECIFIED: Not used
      PRINCIPAL_STATE_MATCHED: Principal matches the GcpUserAccessBinding
        principal.
      PRINCIPAL_STATE_UNMATCHED: Principal does not match the
        GcpUserAccessBinding principal.
      PRINCIPAL_STATE_NOT_FOUND: GcpUserAccessBinding principal does not
        exist.
      PRINCIPAL_STATE_INFO_DENIED: Principal does not have enough permission
        to read the GcpUserAccessBinding principal.
      PRINCIPAL_STATE_UNSUPPORTED: Denied or target principal is not supported
        to troubleshoot.
    """
    PRINCIPAL_STATE_UNSPECIFIED = 0
    PRINCIPAL_STATE_MATCHED = 1
    PRINCIPAL_STATE_UNMATCHED = 2
    PRINCIPAL_STATE_NOT_FOUND = 3
    PRINCIPAL_STATE_INFO_DENIED = 4
    PRINCIPAL_STATE_UNSUPPORTED = 5