from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityFlavorsValueListEntryValuesEnum(_messages.Enum):
    """SecurityFlavorsValueListEntryValuesEnum enum type.

    Values:
      SECURITY_FLAVOR_UNSPECIFIED: SecurityFlavor not set.
      AUTH_SYS: The user's UNIX user-id and group-ids are transferred "in the
        clear" (not encrypted) on the network, unauthenticated by the NFS
        server (default).
      KRB5: End-user authentication through Kerberos V5.
      KRB5I: krb5 plus integrity protection (data packets are tamper proof).
      KRB5P: krb5i plus privacy protection (data packets are tamper proof and
        encrypted).
    """
    SECURITY_FLAVOR_UNSPECIFIED = 0
    AUTH_SYS = 1
    KRB5 = 2
    KRB5I = 3
    KRB5P = 4