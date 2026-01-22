from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceInstance(_messages.Message):
    """Message describing ServiceInstance object TODO(user) add appropriate
  visibility tags to the fields of this proto.

  Enums:
    RequestedStateValueValuesEnum: Output only. The intended state to which
      the service instance is reconciling.
    StateValueValuesEnum: Output only. The current state.

  Messages:
    AnnotationsValue: Optional. The annotations to associate with this service
      instance. Annotations may be used to store client information, but are
      not used by the server.
    LabelsValue: Optional. The labels to associate with this service instance.
      Labels may be used for filtering and billing tracking.

  Fields:
    annotations: Optional. The annotations to associate with this service
      instance. Annotations may be used to store client information, but are
      not used by the server.
    auxiliaryServicesConfig: Optional. Maintenance policy for this service
      instance. TODO this might end up being a separate API instead of
      inlined. Not in scope for private GA MaintenancePolicy
      maintenance_policy = 19; Configuration of auxiliary services used by
      this instance.
    createTime: Output only. The timestamp when the resource was created.
    displayName: Optional. User-provided human-readable name to be used in
      user interfaces.
    gdceCluster: Optional. A GDCE cluster.
    labels: Optional. The labels to associate with this service instance.
      Labels may be used for filtering and billing tracking.
    name: The name of the service instance.
    reconciling: Output only. Whether the service instance is currently
      reconciling. True if the current state of the resource does not match
      the intended state, and the system is working to reconcile them, whether
      or not the change was user initiated. Required by
      aip.dev/128#reconciliation
    requestedState: Output only. The intended state to which the service
      instance is reconciling.
    sparkServiceInstanceConfig: Optional. Spark-specific service instance
      configuration.
    state: Output only. The current state.
    stateMessage: Output only. A message explaining the current state.
    uid: Output only. System generated unique identifier for this service
      instance, formatted as UUID4.
    updateTime: Output only. The timestamp when the resource was most recently
      updated.
  """

    class RequestedStateValueValuesEnum(_messages.Enum):
        """Output only. The intended state to which the service instance is
    reconciling.

    Values:
      STATE_UNSPECIFIED: The service instance state is unknown.
      CREATING: The service instance is being created and is not yet ready to
        accept requests.
      ACTIVE: The service instance is running.
      DISCONNECTED: The service instance is running but disconnected from the
        Google network
      DELETING: The service instance is being deleted.
      STOPPING: The service instance is being stopped. Not in scope for
        private GA
      STOPPED: The service instance is stopped. Not in scope for private GA
      STARTING: The service instance is being started from being STOPPED. Not
        in scope for private GA
      UPDATING: The service instance is being updated
      FAILED: The service instance has encountered an unrecoverable error.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DISCONNECTED = 3
        DELETING = 4
        STOPPING = 5
        STOPPED = 6
        STARTING = 7
        UPDATING = 8
        FAILED = 9

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state.

    Values:
      STATE_UNSPECIFIED: The service instance state is unknown.
      CREATING: The service instance is being created and is not yet ready to
        accept requests.
      ACTIVE: The service instance is running.
      DISCONNECTED: The service instance is running but disconnected from the
        Google network
      DELETING: The service instance is being deleted.
      STOPPING: The service instance is being stopped. Not in scope for
        private GA
      STOPPED: The service instance is stopped. Not in scope for private GA
      STARTING: The service instance is being started from being STOPPED. Not
        in scope for private GA
      UPDATING: The service instance is being updated
      FAILED: The service instance has encountered an unrecoverable error.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DISCONNECTED = 3
        DELETING = 4
        STOPPING = 5
        STOPPED = 6
        STARTING = 7
        UPDATING = 8
        FAILED = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. The annotations to associate with this service instance.
    Annotations may be used to store client information, but are not used by
    the server.

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
        """Optional. The labels to associate with this service instance. Labels
    may be used for filtering and billing tracking.

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
    auxiliaryServicesConfig = _messages.MessageField('AuxiliaryServicesConfig', 2)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    gdceCluster = _messages.MessageField('GdceCluster', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    reconciling = _messages.BooleanField(8)
    requestedState = _messages.EnumField('RequestedStateValueValuesEnum', 9)
    sparkServiceInstanceConfig = _messages.MessageField('SparkServiceInstanceConfig', 10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    stateMessage = _messages.StringField(12)
    uid = _messages.StringField(13)
    updateTime = _messages.StringField(14)