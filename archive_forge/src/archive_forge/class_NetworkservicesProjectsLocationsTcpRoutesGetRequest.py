from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsTcpRoutesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsTcpRoutesGetRequest object.

  Fields:
    name: Required. A name of the TcpRoute to get. Must be in the format
      `projects/*/locations/global/tcpRoutes/*`.
  """
    name = _messages.StringField(1, required=True)