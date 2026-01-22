from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamInstance(_messages.Message):
    """Message describing StreamInstance object Next ID: 16

  Messages:
    LabelsValue: Labels as key value pairs
    LocationConfigsValue: Deployment configuration of the instance by
      locations (only regions are supported now). Map keys are regions in the
      string form.

  Fields:
    apiEndpoint: Output only. The API endpoint to which an Stream client can
      connect to request a streaming session.
    apiKey: Output only. The API key that an Stream client must use when
      requesting a streaming session.
    content: The content that this instance serves.
    contentBuildVersion: The user-specified version tag and build ID of the
      content served.
    createTime: Output only. [Output only] Create time stamp
    gpuClass: Immutable. The GPU class this instance uses. Default value is
      "general_purpose".
    labels: Labels as key value pairs
    lifecycleState: Output only. Current status of the instance.
    locationConfigs: Deployment configuration of the instance by locations
      (only regions are supported now). Map keys are regions in the string
      form.
    mode: Optional. The XR mode this instance supports. Default value is "ar"
      which supports both 3D and AR experiences.
    name: Identifier. name of resource
    realmConfigs: Deployment configuration of the instance in realms. Note
      that this is not defined as a map for enum types (Realm) cannot be used
      as key.
    streamConfig: Optional. An optional config data to configure the client
      UI.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocationConfigsValue(_messages.Message):
        """Deployment configuration of the instance by locations (only regions
    are supported now). Map keys are regions in the string form.

    Messages:
      AdditionalProperty: An additional property for a LocationConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LocationConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocationConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LocationConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('LocationConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    apiEndpoint = _messages.StringField(1)
    apiKey = _messages.StringField(2)
    content = _messages.StringField(3)
    contentBuildVersion = _messages.MessageField('BuildVersion', 4)
    createTime = _messages.StringField(5)
    gpuClass = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    lifecycleState = _messages.MessageField('LifecycleState', 8)
    locationConfigs = _messages.MessageField('LocationConfigsValue', 9)
    mode = _messages.StringField(10)
    name = _messages.StringField(11)
    realmConfigs = _messages.MessageField('RealmConfig', 12, repeated=True)
    streamConfig = _messages.MessageField('StreamConfig', 13)
    updateTime = _messages.StringField(14)