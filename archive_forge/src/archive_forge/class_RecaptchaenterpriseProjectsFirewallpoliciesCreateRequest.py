from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1FirewallPolicy: A
      GoogleCloudRecaptchaenterpriseV1FirewallPolicy resource to be passed as
      the request body.
    parent: Required. The name of the project this policy will apply to, in
      the format `projects/{project}`.
  """
    googleCloudRecaptchaenterpriseV1FirewallPolicy = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallPolicy', 1)
    parent = _messages.StringField(2, required=True)