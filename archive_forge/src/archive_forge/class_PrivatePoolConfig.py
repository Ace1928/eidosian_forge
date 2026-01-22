from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatePoolConfig(_messages.Message):
    """Configuration for a PrivatePool.

  Enums:
    PrivilegedModeValueValuesEnum: Immutable. Specifies the privileged mode
      for the worker pool. Once created, this setting cannot be changed on the
      worker pool, as we are unable to guarantee that the cluster has not been
      altered by misuse of privileged Docker daemon.

  Messages:
    LoggingSasValue: Output only.

  Fields:
    loggingSas: Output only.
    networkConfig: Network configuration for the pool.
    privilegedMode: Immutable. Specifies the privileged mode for the worker
      pool. Once created, this setting cannot be changed on the worker pool,
      as we are unable to guarantee that the cluster has not been altered by
      misuse of privileged Docker daemon.
    scalingConfig: Configuration options for worker pool.
    securityConfig: Security configuration for the pool.
    workerConfig: Configuration options for individual workers.
    workerPoolGroup: Output only. UUID representing worker pools with the same
      region, privilege mode and network config.
  """

    class PrivilegedModeValueValuesEnum(_messages.Enum):
        """Immutable. Specifies the privileged mode for the worker pool. Once
    created, this setting cannot be changed on the worker pool, as we are
    unable to guarantee that the cluster has not been altered by misuse of
    privileged Docker daemon.

    Values:
      PRIVILEGED_MODE_UNSPECIFIED: Unspecified - this is treated as
        NON_PRIVILEGED_ONLY.
      NON_PRIVILEGED_ONLY: Users can only run builds using a non-privileged
        Docker daemon. This is suitable for most cases.
      PRIVILEGED_PERMITTED: Users are allowed to run builds using a privileged
        Docker daemon. This setting should be used with caution, as using a
        privileged Docker daemon introduces a security risk. A user would want
        this if they need to run "docker-in-docker", i.e. their builds use
        docker or docker-compose.
    """
        PRIVILEGED_MODE_UNSPECIFIED = 0
        NON_PRIVILEGED_ONLY = 1
        PRIVILEGED_PERMITTED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LoggingSasValue(_messages.Message):
        """Output only.

    Messages:
      AdditionalProperty: An additional property for a LoggingSasValue object.

    Fields:
      additionalProperties: Additional properties of type LoggingSasValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LoggingSasValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    loggingSas = _messages.MessageField('LoggingSasValue', 1)
    networkConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV1NetworkConfig', 2)
    privilegedMode = _messages.EnumField('PrivilegedModeValueValuesEnum', 3)
    scalingConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV1ScalingConfig', 4)
    securityConfig = _messages.MessageField('SecurityConfig', 5)
    workerConfig = _messages.MessageField('GoogleDevtoolsCloudbuildV1PrivatePoolConfigWorkerConfig', 6)
    workerPoolGroup = _messages.StringField(7)