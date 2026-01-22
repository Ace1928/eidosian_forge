from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class KmsinventoryProjectsLocationsKeyRingsCryptoKeysGetProtectedResourcesSummaryRequest(_messages.Message):
    """A KmsinventoryProjectsLocationsKeyRingsCryptoKeysGetProtectedResourcesSu
  mmaryRequest object.

  Fields:
    name: Required. The resource name of the CryptoKey.
  """
    name = _messages.StringField(1, required=True)