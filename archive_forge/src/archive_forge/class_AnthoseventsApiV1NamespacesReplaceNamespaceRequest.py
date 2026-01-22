from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesReplaceNamespaceRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesReplaceNamespaceRequest object.

  Fields:
    name: Required. The name of the namespace being retrieved. If needed,
      replace {namespace_id} with the project ID.
    namespace: A Namespace resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    namespace = _messages.MessageField('Namespace', 2)