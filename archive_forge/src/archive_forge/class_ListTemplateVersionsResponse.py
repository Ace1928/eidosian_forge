from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTemplateVersionsResponse(_messages.Message):
    """Respond a list of TemplateVersions.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    templateVersions: A list of TemplateVersions.
  """
    nextPageToken = _messages.StringField(1)
    templateVersions = _messages.MessageField('TemplateVersion', 2, repeated=True)