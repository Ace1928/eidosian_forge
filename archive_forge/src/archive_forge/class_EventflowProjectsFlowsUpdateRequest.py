from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EventflowProjectsFlowsUpdateRequest(_messages.Message):
    """A EventflowProjectsFlowsUpdateRequest object.

  Fields:
    flow: A Flow resource to be passed as the request body.
    name: Name must be unique within a namespace. Is required when creating
      resources, although some resources may allow a client to request the
      generation of an appropriate name automatically. Name is primarily
      intended for creation idempotence and configuration definition. Cannot
      be updated. More info: http://kubernetes.io/docs/user-
      guide/identifiers#names +optional
    namespace: Namespace defines the space within each name must be unique. An
      empty namespace is equivalent to the "default" namespace, but "default"
      is the canonical representation. Not all objects are required to be
      scoped to a namespace - the value of this field for those objects will
      be empty. Must be a DNS_LABEL. Cannot be updated. More info:
      http://kubernetes.io/docs/user-guide/namespaces +optional
  """
    flow = _messages.MessageField('Flow', 1)
    name = _messages.StringField(2, required=True)
    namespace = _messages.StringField(3, required=True)