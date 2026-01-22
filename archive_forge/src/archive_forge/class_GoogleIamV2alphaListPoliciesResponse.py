from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2alphaListPoliciesResponse(_messages.Message):
    """Response message for ListPolicies method.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    policies: The collection of policy metadata that are attached on the
      resource.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('GoogleIamV2alphaPolicy', 2, repeated=True)