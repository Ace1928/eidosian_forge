from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslManagementTypeValueValuesEnum(_messages.Enum):
    """SSL management type for this domain. If AUTOMATIC, a managed
    certificate is automatically provisioned. If MANUAL, certificate_id must
    be manually specified in order to configure SSL for this domain.

    Values:
      AUTOMATIC: SSL support for this domain is configured automatically. The
        mapped SSL certificate will be automatically renewed.
      MANUAL: SSL support for this domain is configured manually by the user.
        Either the domain has no SSL support or a user-obtained SSL
        certificate has been explictly mapped to this domain.
    """
    AUTOMATIC = 0
    MANUAL = 1