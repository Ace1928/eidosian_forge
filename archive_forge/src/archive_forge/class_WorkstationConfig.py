from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationConfig(_messages.Message):
    """A workstation configuration resource in the Cloud Workstations API.
  Workstation configurations act as templates for workstations. The
  workstation configuration defines details such as the workstation virtual
  machine (VM) instance type, persistent storage, container image defining
  environment, which IDE or Code Editor to use, and more. Administrators and
  platform teams can also use [Identity and Access Management
  (IAM)](https://cloud.google.com/iam/docs/overview) rules to grant access to
  teams or to individual developers.

  Messages:
    AnnotationsValue: Optional. Client-specified annotations.
    LabelsValue: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation configuration and that are also
      propagated to the underlying Compute Engine resources.

  Fields:
    annotations: Optional. Client-specified annotations.
    conditions: Output only. Status conditions describing the current resource
      state.
    container: Optional. Container that runs upon startup for each workstation
      using this workstation configuration.
    createTime: Output only. Time when this workstation configuration was
      created.
    degraded: Output only. Whether this resource is degraded, in which case it
      may require user action to restore full functionality. See also the
      conditions field.
    deleteTime: Output only. Time when this workstation configuration was
      soft-deleted.
    disableTcpConnections: Optional. Disables support for plain TCP
      connections in the workstation. By default the service supports TCP
      connections through a websocket relay. Setting this option to true
      disables that relay, which prevents the usage of services that require
      plain TCP connections, such as SSH. When enabled, all communication must
      occur over HTTPS or WSS.
    displayName: Optional. Human-readable name for this workstation
      configuration.
    enableAuditAgent: Optional. Whether to enable Linux `auditd` logging on
      the workstation. When enabled, a service account must also be specified
      that has `logging.buckets.write` permission on the project. Operating
      system audit logging is distinct from [Cloud Audit
      Logs](https://cloud.google.com/workstations/docs/audit-logging).
    encryptionKey: Immutable. Encrypts resources of this workstation
      configuration using a customer-managed encryption key (CMEK). If
      specified, the boot disk of the Compute Engine instance and the
      persistent disk are encrypted using this encryption key. If this field
      is not set, the disks are encrypted using a generated key. Customer-
      managed encryption keys do not protect disk metadata. If the customer-
      managed encryption key is rotated, when the workstation instance is
      stopped, the system attempts to recreate the persistent disk with the
      new version of the key. Be sure to keep older versions of the key until
      the persistent disk is recreated. Otherwise, data on the persistent disk
      might be lost. If the encryption key is revoked, the workstation session
      automatically stops within 7 hours. Immutable after the workstation
      configuration is created.
    ephemeralDirectories: Optional. Ephemeral directories which won't persist
      across workstation sessions.
    etag: Optional. Checksum computed by the server. May be sent on update and
      delete requests to make sure that the client has an up-to-date value
      before proceeding.
    host: Optional. Runtime host for the workstation.
    idleTimeout: Optional. Number of seconds to wait before automatically
      stopping a workstation after it last received user traffic. A value of
      `"0s"` indicates that Cloud Workstations VMs created with this
      configuration should never time out due to idleness. Provide
      [duration](https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#duration) terminated by `s` for
      seconds-for example, `"7200s"` (2 hours). The default is `"1200s"` (20
      minutes).
    labels: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation configuration and that are also
      propagated to the underlying Compute Engine resources.
    name: Identifier. Full name of this workstation configuration.
    persistentDirectories: Optional. Directories to persist across workstation
      sessions.
    readinessChecks: Optional. Readiness checks to perform when starting a
      workstation using this workstation configuration. Mark a workstation as
      running only after all specified readiness checks return 200 status
      codes.
    reconciling: Output only. Indicates whether this workstation configuration
      is currently being updated to match its intended state.
    replicaZones: Optional. Immutable. Specifies the zones used to replicate
      the VM and disk resources within the region. If set, exactly two zones
      within the workstation cluster's region must be specified-for example,
      `['us-central1-a', 'us-central1-f']`. If this field is empty, two
      default zones within the region are used. Immutable after the
      workstation configuration is created.
    runningTimeout: Optional. Number of seconds that a workstation can run
      until it is automatically shut down. We recommend that workstations be
      shut down daily to reduce costs and so that security updates can be
      applied upon restart. The idle_timeout and running_timeout fields are
      independent of each other. Note that the running_timeout field shuts
      down VMs after the specified time, regardless of whether or not the VMs
      are idle. Provide duration terminated by `s` for seconds-for example,
      `"54000s"` (15 hours). Defaults to `"43200s"` (12 hours). A value of
      `"0s"` indicates that workstations using this configuration should never
      time out. If encryption_key is set, it must be greater than `"0s"` and
      less than `"86400s"` (24 hours). Warning: A value of `"0s"` indicates
      that Cloud Workstations VMs created with this configuration have no
      maximum running time. This is strongly discouraged because you incur
      costs and will not pick up security updates.
    uid: Output only. A system-assigned unique identifier for this workstation
      configuration.
    updateTime: Output only. Time when this workstation configuration was most
      recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Client-specified annotations.

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
        """Optional. [Labels](https://cloud.google.com/workstations/docs/label-
    resources) that are applied to the workstation configuration and that are
    also propagated to the underlying Compute Engine resources.

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
    conditions = _messages.MessageField('Status', 2, repeated=True)
    container = _messages.MessageField('Container', 3)
    createTime = _messages.StringField(4)
    degraded = _messages.BooleanField(5)
    deleteTime = _messages.StringField(6)
    disableTcpConnections = _messages.BooleanField(7)
    displayName = _messages.StringField(8)
    enableAuditAgent = _messages.BooleanField(9)
    encryptionKey = _messages.MessageField('CustomerEncryptionKey', 10)
    ephemeralDirectories = _messages.MessageField('EphemeralDirectory', 11, repeated=True)
    etag = _messages.StringField(12)
    host = _messages.MessageField('Host', 13)
    idleTimeout = _messages.StringField(14)
    labels = _messages.MessageField('LabelsValue', 15)
    name = _messages.StringField(16)
    persistentDirectories = _messages.MessageField('PersistentDirectory', 17, repeated=True)
    readinessChecks = _messages.MessageField('ReadinessCheck', 18, repeated=True)
    reconciling = _messages.BooleanField(19)
    replicaZones = _messages.StringField(20, repeated=True)
    runningTimeout = _messages.StringField(21)
    uid = _messages.StringField(22)
    updateTime = _messages.StringField(23)