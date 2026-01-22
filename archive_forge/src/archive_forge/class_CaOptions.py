from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CaOptions(_messages.Message):
    """Describes values that are relevant in a CA certificate.

  Fields:
    isCa: Optional. Refers to the "CA" X.509 extension, which is a boolean
      value. When this value is missing, the extension will be omitted from
      the CA certificate.
    maxIssuerPathLength: Optional. Refers to the path length restriction X.509
      extension. For a CA certificate, this value describes the depth of
      subordinate CA certificates that are allowed. If this value is less than
      0, the request will fail. If this value is missing, the max path length
      will be omitted from the CA certificate.
  """
    isCa = _messages.BooleanField(1)
    maxIssuerPathLength = _messages.IntegerField(2, variant=_messages.Variant.INT32)