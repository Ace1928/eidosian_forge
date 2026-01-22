from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerLiensListRequest(_messages.Message):
    """A CloudresourcemanagerLiensListRequest object.

  Fields:
    pageSize: The maximum number of items to return. This is a suggestion for
      the server. The server can return fewer liens than requested. If
      unspecified, server picks an appropriate default.
    pageToken: The `next_page_token` value returned from a previous List
      request, if any.
    parent: Required. The name of the resource to list all attached Liens. For
      example, `projects/1234`. (google.api.field_policy).resource_type
      annotation is not set since the parent depends on the meta api
      implementation. This field could be a project or other sub project
      resources.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)