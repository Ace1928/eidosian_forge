from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TlsInfoCommonName(_messages.Message):
    """A GoogleCloudApigeeV1TlsInfoCommonName object.

  Fields:
    value: The TLS Common Name string of the certificate.
    wildcardMatch: Indicates whether the cert should be matched against as a
      wildcard cert.
  """
    value = _messages.StringField(1)
    wildcardMatch = _messages.BooleanField(2)