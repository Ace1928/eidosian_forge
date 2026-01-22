from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IssuanceModes(_messages.Message):
    """IssuanceModes specifies the allowed ways in which Certificates may be
  requested from this CaPool.

  Fields:
    allowConfigBasedIssuance: Optional. When true, allows callers to create
      Certificates by specifying a CertificateConfig.
    allowCsrBasedIssuance: Optional. When true, allows callers to create
      Certificates by specifying a CSR.
  """
    allowConfigBasedIssuance = _messages.BooleanField(1)
    allowCsrBasedIssuance = _messages.BooleanField(2)