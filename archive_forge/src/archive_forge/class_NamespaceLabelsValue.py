from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NamespaceLabelsValue(_messages.Message):
    """Optional. Scope-level cluster namespace labels. For the member
    clusters bound to the Scope, these labels are applied to each namespace
    under the Scope. Scope-level labels take precedence over Namespace-level
    labels (`namespace_labels` in the Fleet Namespace resource) if they share
    a key. Keys and values must be Kubernetes-conformant.

    Messages:
      AdditionalProperty: An additional property for a NamespaceLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type NamespaceLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NamespaceLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)