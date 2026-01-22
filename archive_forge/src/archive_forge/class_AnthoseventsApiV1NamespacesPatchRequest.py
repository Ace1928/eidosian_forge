from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesPatchRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesPatchRequest object.

  Fields:
    name: Required. The name of the namespace being retrieved. If needed,
      replace {namespace_id} with the project ID.
    namespace: A Namespace resource to be passed as the request body.
    updateMask: Required. Indicates which fields in the provided namespace to
      update. This field is currently unused.
  """
    name = _messages.StringField(1, required=True)
    namespace = _messages.MessageField('Namespace', 2)
    updateMask = _messages.StringField(3)