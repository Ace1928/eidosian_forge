from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTerraformVersionsResponse(_messages.Message):
    """The response message for the `ListTerraformVersions` method.

  Fields:
    nextPageToken: Token to be supplied to the next ListTerraformVersions
      request via `page_token` to obtain the next set of results.
    terraformVersions: List of TerraformVersions.
    unreachable: Unreachable resources, if any.
  """
    nextPageToken = _messages.StringField(1)
    terraformVersions = _messages.MessageField('TerraformVersion', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)