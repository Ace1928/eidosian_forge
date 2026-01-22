from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsClientTlsPoliciesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsClientTlsPoliciesGetRequest object.

  Fields:
    name: Required. A name of the ClientTlsPolicy to get. Must be in the
      format `projects/*/locations/{location}/clientTlsPolicies/*`.
  """
    name = _messages.StringField(1, required=True)