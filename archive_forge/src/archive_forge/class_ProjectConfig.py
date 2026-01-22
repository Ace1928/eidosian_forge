from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectConfig(_messages.Message):
    """Cloud Source Repositories configuration of a project.

  Messages:
    PubsubConfigsValue: How this project publishes a change in the
      repositories through Cloud Pub/Sub. Keyed by the topic names.

  Fields:
    enablePrivateKeyCheck: Reject a Git push that contains a private key.
    name: The name of the project. Values are of the form `projects/`.
    pubsubConfigs: How this project publishes a change in the repositories
      through Cloud Pub/Sub. Keyed by the topic names.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PubsubConfigsValue(_messages.Message):
        """How this project publishes a change in the repositories through Cloud
    Pub/Sub. Keyed by the topic names.

    Messages:
      AdditionalProperty: An additional property for a PubsubConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type PubsubConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PubsubConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A PubsubConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PubsubConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    enablePrivateKeyCheck = _messages.BooleanField(1)
    name = _messages.StringField(2)
    pubsubConfigs = _messages.MessageField('PubsubConfigsValue', 3)