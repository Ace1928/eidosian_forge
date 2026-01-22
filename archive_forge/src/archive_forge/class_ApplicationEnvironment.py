from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationEnvironment(_messages.Message):
    """Represents the ApplicationEnvironment resource.

  Messages:
    AnnotationsValue: Optional. The annotations to associate with this
      application environment. Annotations may be used to store client
      information, but are not used by the server.
    LabelsValue: Optional. The labels to associate with this application
      environment. Labels may be used for filtering and billing tracking.

  Fields:
    annotations: Optional. The annotations to associate with this application
      environment. Annotations may be used to store client information, but
      are not used by the server.
    createTime: Output only. The timestamp when the resource was created.
    displayName: Optional. User-provided human-readable name to be used in
      user interfaces.
    labels: Optional. The labels to associate with this application
      environment. Labels may be used for filtering and billing tracking.
    name: Identifier. Fields 1-6 should exist for all declarative friendly
      resources per aip.dev/148 The name of the application environment.
      Format: projects/{project}/locations/{location}/serviceInstances/{servic
      e_instance}/applicationEnvironments/{application_environment_id}
    namespace: Optional. The name of the namespace in which to create this
      ApplicationEnvironment. This namespace must already exist in the cluster
    sparkApplicationEnvironmentConfig: Optional. The engine-specific
      configurations for this ApplicationEnvironment.
    uid: Output only. System generated unique identifier for this application
      environment, formatted as UUID4.
    updateTime: Output only. The timestamp when the resource was most recently
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. The annotations to associate with this application
    environment. Annotations may be used to store client information, but are
    not used by the server.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this application environment.
    Labels may be used for filtering and billing tracking.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    namespace = _messages.StringField(6)
    sparkApplicationEnvironmentConfig = _messages.MessageField('SparkApplicationEnvironmentConfig', 7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)