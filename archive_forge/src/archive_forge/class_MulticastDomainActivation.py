from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastDomainActivation(_messages.Message):
    """Multicast domain activation resource.

  Messages:
    LabelsValue: Labels as key-value pairs

  Fields:
    adminNetwork: Output only. [Output only] The URL of the admin network.
    createTime: Output only. [Output only] The timestamp when the multicast
      domain activation was created.
    domain: Reference to the domain that is being activated. [Deprecated] Use
      multicast_domain instead.
    labels: Labels as key-value pairs
    multicastDomain: Optional. The resource name of the multicast domain to
      activate. Use the following format:
      `projects/*/locations/global/multicastDomains/*`.
    name: The resource name of the multicast domain activation. Use the
      following format: `projects/*/locations/*/multicastDomainActivations/*`.
    updateTime: Output only. [Output only] The timestamp when the multicast
      domain activation was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key-value pairs

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
    adminNetwork = _messages.StringField(1)
    createTime = _messages.StringField(2)
    domain = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    multicastDomain = _messages.StringField(5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)