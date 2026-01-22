from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest(_messages.Message):
    """The reorder firewall policies request message.

  Fields:
    names: Required. A list containing all policy names, in the new order.
      Each name is in the format
      `projects/{project}/firewallpolicies/{firewallpolicy}`.
  """
    names = _messages.StringField(1, repeated=True)