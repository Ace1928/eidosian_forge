from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsTlsInspectionPoliciesCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsTlsInspectionPoliciesCreateRequest
  object.

  Fields:
    parent: Required. The parent resource of the TlsInspectionPolicy. Must be
      in the format `projects/{project}/locations/{location}`.
    tlsInspectionPolicy: A TlsInspectionPolicy resource to be passed as the
      request body.
    tlsInspectionPolicyId: Required. Short name of the TlsInspectionPolicy
      resource to be created. This value should be 1-63 characters long,
      containing only letters, numbers, hyphens, and underscores, and should
      not start with a number. E.g. "tls_inspection_policy1".
  """
    parent = _messages.StringField(1, required=True)
    tlsInspectionPolicy = _messages.MessageField('TlsInspectionPolicy', 2)
    tlsInspectionPolicyId = _messages.StringField(3)