from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateAuthorityConfig(_messages.Message):
    """The CA that issues the workload certificate. It includes CA address,
  type, authentication to CA service, etc.

  Fields:
    certificateAuthorityServiceConfig: Defines a
      CertificateAuthorityServiceConfig.
  """
    certificateAuthorityServiceConfig = _messages.MessageField('CertificateAuthorityServiceConfig', 1)