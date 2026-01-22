from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamFoldersLocationsPolicyBindingsListRequest(_messages.Message):
    """A IamFoldersLocationsPolicyBindingsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      Filter rules are case insensitive. Some eligible fields for filtering
      are: + `target` + `policy` Some examples of filter queries: | Query |
      Description | |------------------|--------------------------------------
      ---------------| | target:how* | The binding target's name starts with
      "how ". | | target:howl | The binding target's name is `howl`. | |
      policy:howl | The binding policy's name is 'howl'. |
    pageSize: Optional. The maximum number of policy bindings to return. The
      service may return fewer than this value. If unspecified, at most 50
      policy bindings will be returned. The maximum value is 1000; values
      above 1000 will be coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListPolicyBindings` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListPolicyBindings`
      must match the call that provided the page token.
    parent: Required. The parent resource, which owns the collection of policy
      bindings. Format: `projects/{project_id}/locations/{location}`
      `projects/{project_number}/locations/{location}`
      `folders/{folder_id}/locations/{location}`
      `organizations/{organization_id}/locations/{location}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)