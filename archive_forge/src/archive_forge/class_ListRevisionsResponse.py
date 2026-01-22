from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRevisionsResponse(_messages.Message):
    """ListRevisionsResponse is a list of Revision resources.

  Fields:
    apiVersion: The API version for this call such as
      "serving.knative.dev/v1".
    items: List of Revisions.
    kind: The kind of this resource, in this case "RevisionList".
    metadata: Metadata associated with this revision list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('Revision', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)