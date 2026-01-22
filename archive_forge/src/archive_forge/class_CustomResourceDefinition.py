from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceDefinition(_messages.Message):
    """CustomResourceDefinition represents a resource that should be exposed on
  the API server. Its name MUST be in the format <.spec.name>.<.spec.group>.

  Fields:
    apiVersion: The API version for this call such as
      "k8s.apiextensions.io/v1".
    kind: The kind of resource, in this case always
      "CustomResourceDefinition".
    metadata: Metadata associated with this CustomResourceDefinition.
    spec: Spec describes how the user wants the resources to appear
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('CustomResourceDefinitionSpec', 4)