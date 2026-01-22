from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentKnowledgeBasesDocumentsListRequest(_messages.Message):
    """A DialogflowProjectsAgentKnowledgeBasesDocumentsListRequest object.

  Fields:
    filter: The filter expression used to filter documents returned by the
      list method. The expression has the following syntax: [AND ] ... The
      following fields and operators are supported: * knowledge_types with
      has(:) operator * display_name with has(:) operator * state with
      equals(=) operator Examples: * "knowledge_types:FAQ" matches documents
      with FAQ knowledge type. * "display_name:customer" matches documents
      whose display name contains "customer". * "state=ACTIVE" matches
      documents with ACTIVE state. * "knowledge_types:FAQ AND state=ACTIVE"
      matches all active FAQ documents. For more information about filtering,
      see [API Filtering](https://aip.dev/160).
    pageSize: The maximum number of items to return in a single page. By
      default 10 and at most 100.
    pageToken: The next_page_token value returned from a previous list
      request.
    parent: Required. The knowledge base to list all documents for. Format:
      `projects//locations//knowledgeBases/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)