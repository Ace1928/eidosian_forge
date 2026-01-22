from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsFirewallpoliciesListRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsFirewallpoliciesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of policies to return. Default is
      10. Max limit is 1000.
    pageToken: Optional. The next_page_token value returned from a previous.
      ListFirewallPoliciesRequest, if any.
    parent: Required. The name of the project to list the policies for, in the
      format `projects/{project}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)