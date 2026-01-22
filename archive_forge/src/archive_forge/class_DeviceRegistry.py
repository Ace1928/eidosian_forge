from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeviceRegistry(_messages.Message):
    """A container for a group of devices.

  Enums:
    LogLevelValueValuesEnum: **Beta Feature** The default logging verbosity
      for activity from devices in this registry. The verbosity level can be
      overridden by Device.log_level.

  Fields:
    credentials: The credentials used to verify the device credentials. No
      more than 10 credentials can be bound to a single registry at a time.
      The verification process occurs at the time of device creation or
      update. If this field is empty, no verification is performed. Otherwise,
      the credentials of a newly created device or added credentials of an
      updated device should be signed with one of these registry credentials.
      Note, however, that existing devices will never be affected by
      modifications to this list of credentials: after a device has been
      successfully created in a registry, it should be able to connect even if
      its registry credentials are revoked, deleted, or modified.
    eventNotificationConfigs: The configuration for notification of telemetry
      events received from the device. All telemetry events that were
      successfully published by the device and acknowledged by Cloud IoT Core
      are guaranteed to be delivered to Cloud Pub/Sub. If multiple
      configurations match a message, only the first matching configuration is
      used. If you try to publish a device telemetry event using MQTT without
      specifying a Cloud Pub/Sub topic for the device's registry, the
      connection closes automatically. If you try to do so using an HTTP
      connection, an error is returned. Up to 10 configurations may be
      provided.
    httpConfig: The DeviceService (HTTP) configuration for this device
      registry.
    id: The identifier of this device registry. For example, `myRegistry`.
    logLevel: **Beta Feature** The default logging verbosity for activity from
      devices in this registry. The verbosity level can be overridden by
      Device.log_level.
    mqttConfig: The MQTT configuration for this device registry.
    name: The resource path name. For example, `projects/example-
      project/locations/us-central1/registries/my-registry`.
    stateNotificationConfig: The configuration for notification of new states
      received from the device. State updates are guaranteed to be stored in
      the state history, but notifications to Cloud Pub/Sub are not
      guaranteed. For example, if permissions are misconfigured or the
      specified topic doesn't exist, no notification will be published but the
      state will still be stored in Cloud IoT Core.
  """

    class LogLevelValueValuesEnum(_messages.Enum):
        """**Beta Feature** The default logging verbosity for activity from
    devices in this registry. The verbosity level can be overridden by
    Device.log_level.

    Values:
      LOG_LEVEL_UNSPECIFIED: No logging specified. If not specified, logging
        will be disabled.
      NONE: Disables logging.
      ERROR: Error events will be logged.
      INFO: Informational events will be logged, such as connections and
        disconnections.
      DEBUG: All events will be logged.
    """
        LOG_LEVEL_UNSPECIFIED = 0
        NONE = 1
        ERROR = 2
        INFO = 3
        DEBUG = 4
    credentials = _messages.MessageField('RegistryCredential', 1, repeated=True)
    eventNotificationConfigs = _messages.MessageField('EventNotificationConfig', 2, repeated=True)
    httpConfig = _messages.MessageField('HttpConfig', 3)
    id = _messages.StringField(4)
    logLevel = _messages.EnumField('LogLevelValueValuesEnum', 5)
    mqttConfig = _messages.MessageField('MqttConfig', 6)
    name = _messages.StringField(7)
    stateNotificationConfig = _messages.MessageField('StateNotificationConfig', 8)