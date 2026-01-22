from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsTlsRoutesDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsTlsRoutesDeleteRequest object.

  Fields:
    name: Required. A name of the TlsRoute to delete. Must be in the format
      `projects/*/locations/global/tlsRoutes/*`.
  """
    name = _messages.StringField(1, required=True)