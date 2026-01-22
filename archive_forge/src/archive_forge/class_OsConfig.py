from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsConfig(_messages.Message):
    """An OS Config resource describing a set of OS configs that should be set
  on a group of instances.

  Messages:
    LabelsValue: Represents cloud resource labels.

  Fields:
    apt: Optional package manager configurations for apt.
    createTime: Output only. Time this OsConfig was created.
    description: Description of the OsConfig. Length of the description is
      limited to 1024 characters.
    goo: Optional package manager configurations for windows.
    labels: Represents cloud resource labels.
    name: Identifying name for this OsConfig.
    updateTime: Output only. Last time this OsConfig was updated.
    windowsUpdate: Optional Windows Update configurations.
    yum: Optional package manager configurations for yum.
    zypper: Optional package manager configuration for zypper.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Represents cloud resource labels.

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
    apt = _messages.MessageField('AptPackageConfig', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    goo = _messages.MessageField('GooPackageConfig', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)
    windowsUpdate = _messages.MessageField('WindowsUpdateConfig', 8)
    yum = _messages.MessageField('YumPackageConfig', 9)
    zypper = _messages.MessageField('ZypperPackageConfig', 10)