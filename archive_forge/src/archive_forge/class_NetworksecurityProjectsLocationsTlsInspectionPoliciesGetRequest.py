from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsTlsInspectionPoliciesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsTlsInspectionPoliciesGetRequest
  object.

  Fields:
    name: Required. A name of the TlsInspectionPolicy to get. Must be in the
      format `projects/{project}/locations/{location}/tlsInspectionPolicies/{t
      ls_inspection_policy}`.
  """
    name = _messages.StringField(1, required=True)