from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Appliance(_messages.Message):
    """Appliance represents a deployable unit within 3P integration manager.

  Enums:
    TypeValueValuesEnum: Required. The type of the networked Appliance being
      deployed. Used to capture potential differences in deployments between
      types of appliances. Currently only FIREWALL OUT OF BAND deployment is
      supported now, but future values may include SWG (secure web gateway) or
      WAF (web application firewall) with both in-line and Out of band
      deployments.

  Messages:
    AnnotationsValue: Optional. Set of annotations ( ) to allow for custom
      metadata associated with the Appliance resource.

  Fields:
    annotations: Optional. Set of annotations ( ) to allow for custom metadata
      associated with the Appliance resource.
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A text description of the third party appliance
      resource. Might include details such as product features, vendor
      information, and pricing. Character set is UTF-8 and is limited to 4096
      characters.
    displayName: Optional. Name of the third party appliance.
    eulaUri: Optional. A URL to the End User License Agreement provided by the
      third party appliance vendor.
    externalLicenseInfo: Optional. Information about the License type of the
      third party appliance.
    imageVersion: Optional. Version of the Vendor Image of the Firewall VM to
      be deployed, e.g.: 9.1.3
    name: Required. Name of the Appliance resource. It matches the pattern
      `projects/{project}/locations/{location}/appliances/` and must be
      unique.
    partner: Optional. Name of the third party vendor providing the appliance.
    type: Required. The type of the networked Appliance being deployed. Used
      to capture potential differences in deployments between types of
      appliances. Currently only FIREWALL OUT OF BAND deployment is supported
      now, but future values may include SWG (secure web gateway) or WAF (web
      application firewall) with both in-line and Out of band deployments.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the networked Appliance being deployed. Used to
    capture potential differences in deployments between types of appliances.
    Currently only FIREWALL OUT OF BAND deployment is supported now, but
    future values may include SWG (secure web gateway) or WAF (web application
    firewall) with both in-line and Out of band deployments.

    Values:
      TYPE_UNSPECIFIED: Default value.
      OUT_OF_BAND_FIREWALL: A firewall appliance deployed out of band.
      INLINE_FIREWALL: A firewall appliance deployed inline.
    """
        TYPE_UNSPECIFIED = 0
        OUT_OF_BAND_FIREWALL = 1
        INLINE_FIREWALL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Set of annotations ( ) to allow for custom metadata
    associated with the Appliance resource.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    eulaUri = _messages.StringField(5)
    externalLicenseInfo = _messages.StringField(6)
    imageVersion = _messages.StringField(7)
    name = _messages.StringField(8)
    partner = _messages.StringField(9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    updateTime = _messages.StringField(11)