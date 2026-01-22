from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsKnowledgeBasesListRequest(_messages.Message):
    """A DialogflowProjectsLocationsKnowledgeBasesListRequest object.

  Fields:
    filter: The filter expression used to filter knowledge bases returned by
      the list method. The expression has the following syntax: [AND ] ... The
      following fields and operators are supported: * display_name with has(:)
      operator * language_code with equals(=) operator Examples: *
      'language_code=en-us' matches knowledge bases with en-us language code.
      * 'display_name:articles' matches knowledge bases whose display name
      contains "articles". * 'display_name:"Best Articles"' matches knowledge
      bases whose display name contains "Best Articles". * 'language_code=en-
      gb AND display_name=articles' matches all knowledge bases whose display
      name contains "articles" and whose language code is "en-gb". Note: An
      empty filter string (i.e. "") is a no-op and will result in no
      filtering. For more information about filtering, see [API
      Filtering](https://aip.dev/160).
    pageSize: The maximum number of items to return in a single page. By
      default 10 and at most 100.
    pageToken: The next_page_token value returned from a previous list
      request.
    parent: Required. The project to list of knowledge bases for. Format:
      `projects//locations/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)