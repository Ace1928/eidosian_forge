from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneNodeConfig(_messages.Message):
    """BareMetalStandaloneNodeConfig lists machine addresses to access Nodes.

  Messages:
    LabelsValue: The labels assigned to this node. An object containing a list
      of key/value pairs. The labels here, unioned with the labels set on
      BareMetalStandaloneNodePoolConfig are the set of labels that will be
      applied to the node. If there are any conflicts, the
      BareMetalStandaloneNodeConfig labels take precedence. Example: { "name":
      "wrench", "mass": "1.3kg", "count": "3" }.

  Fields:
    labels: The labels assigned to this node. An object containing a list of
      key/value pairs. The labels here, unioned with the labels set on
      BareMetalStandaloneNodePoolConfig are the set of labels that will be
      applied to the node. If there are any conflicts, the
      BareMetalStandaloneNodeConfig labels take precedence. Example: { "name":
      "wrench", "mass": "1.3kg", "count": "3" }.
    nodeIp: The default IPv4 address for SSH access and Kubernetes node.
      Example: 192.168.0.1
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels assigned to this node. An object containing a list of
    key/value pairs. The labels here, unioned with the labels set on
    BareMetalStandaloneNodePoolConfig are the set of labels that will be
    applied to the node. If there are any conflicts, the
    BareMetalStandaloneNodeConfig labels take precedence. Example: { "name":
    "wrench", "mass": "1.3kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    nodeIp = _messages.StringField(2)