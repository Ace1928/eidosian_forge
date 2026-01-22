from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Finding(_messages.Message):
    """Security Command Center finding. A finding is a record of assessment
  data like security, risk, health, or privacy, that is ingested into Security
  Command Center for presentation, notification, analysis, policy testing, and
  enforcement. For example, a cross-site scripting (XSS) vulnerability in an
  App Engine application is a finding.

  Enums:
    FindingClassValueValuesEnum: The class of the finding.
    MuteValueValuesEnum: Indicates the mute state of a finding (either muted,
      unmuted or undefined). Unlike other attributes of a finding, a finding
      provider shouldn't set the value of mute.
    SeverityValueValuesEnum: The severity of the finding. This field is
      managed by the source that writes the finding.
    StateValueValuesEnum: Output only. The state of the finding.

  Messages:
    ContactsValue: Output only. Map containing the points of contact for the
      given finding. The key represents the type of contact, while the value
      contains a list of all the contacts that pertain. Please refer to:
      https://cloud.google.com/resource-manager/docs/managing-notification-
      contacts#notification-categories { "security": { "contacts": [ {
      "email": "person1@company.com" }, { "email": "person2@company.com" } ] }
      }
    ExternalSystemsValue: Output only. Third party SIEM/SOAR fields within
      SCC, contains external system information and external system finding
      fields.
    SourcePropertiesValue: Source specific properties. These properties are
      managed by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.

  Fields:
    access: Access details associated with the finding, such as more
      information on the caller, which method was accessed, and from where.
    application: Represents an application associated with the finding.
    attackExposure: The results of an attack path simulation relevant to this
      finding.
    backupDisasterRecovery: Fields related to Backup and DR findings.
    canonicalName: Output only. The canonical name of the finding. The
      following list shows some examples: + `organizations/{organization_id}/s
      ources/{source_id}/findings/{finding_id}` + `organizations/{organization
      _id}/sources/{source_id}/locations/{location_id}/findings/{finding_id}`
      + `folders/{folder_id}/sources/{source_id}/findings/{finding_id}` + `fol
      ders/{folder_id}/sources/{source_id}/locations/{location_id}/findings/{f
      inding_id}` +
      `projects/{project_id}/sources/{source_id}/findings/{finding_id}` + `pro
      jects/{project_id}/sources/{source_id}/locations/{location_id}/findings/
      {finding_id}` The prefix is the closest CRM ancestor of the resource
      associated with the finding.
    category: Immutable. The additional taxonomy group within findings from a
      given source. Example: "XSS_FLASH_INJECTION"
    cloudDlpDataProfile: Cloud DLP data profile that is associated with the
      finding.
    cloudDlpInspection: Cloud Data Loss Prevention (Cloud DLP) inspection
      results that are associated with the finding.
    compliances: Contains compliance information for security standards
      associated to the finding.
    connections: Contains information about the IP connection associated with
      the finding.
    contacts: Output only. Map containing the points of contact for the given
      finding. The key represents the type of contact, while the value
      contains a list of all the contacts that pertain. Please refer to:
      https://cloud.google.com/resource-manager/docs/managing-notification-
      contacts#notification-categories { "security": { "contacts": [ {
      "email": "person1@company.com" }, { "email": "person2@company.com" } ] }
      }
    containers: Containers associated with the finding. This field provides
      information for both Kubernetes and non-Kubernetes containers.
    createTime: Output only. The time at which the finding was created in
      Security Command Center.
    database: Database associated with the finding.
    description: Contains more details about the finding.
    eventTime: The time the finding was first detected. If an existing finding
      is updated, then this is the time the update occurred. For example, if
      the finding represents an open firewall, this property captures the time
      the detector believes the firewall became open. The accuracy is
      determined by the detector. If the finding is later resolved, then this
      time reflects when the finding was resolved. This must not be set to a
      value greater than the current timestamp.
    exfiltration: Represents exfiltrations associated with the finding.
    externalSystems: Output only. Third party SIEM/SOAR fields within SCC,
      contains external system information and external system finding fields.
    externalUri: The URI that, if available, points to a web page outside of
      Security Command Center where additional information about the finding
      can be found. This field is guaranteed to be either empty or a well
      formed URL.
    files: File associated with the finding.
    findingClass: The class of the finding.
    iamBindings: Represents IAM bindings associated with the finding.
    indicator: Represents what's commonly known as an *indicator of
      compromise* (IoC) in computer forensics. This is an artifact observed on
      a network or in an operating system that, with high confidence,
      indicates a computer intrusion. For more information, see [Indicator of
      compromise](https://en.wikipedia.org/wiki/Indicator_of_compromise).
    kernelRootkit: Signature of the kernel rootkit.
    kubernetes: Kubernetes resources associated with the finding.
    loadBalancers: The load balancers associated with the finding.
    logEntries: Log entries that are relevant to the finding.
    mitreAttack: MITRE ATT&CK tactics and techniques related to this finding.
      See: https://attack.mitre.org
    moduleName: Unique identifier of the module which generated the finding.
      Example: folders/598186756061/securityHealthAnalyticsSettings/customModu
      les/56799441161885
    mute: Indicates the mute state of a finding (either muted, unmuted or
      undefined). Unlike other attributes of a finding, a finding provider
      shouldn't set the value of mute.
    muteInitiator: Records additional information about the mute operation,
      for example, the [mute configuration](https://cloud.google.com/security-
      command-center/docs/how-to-mute-findings) that muted the finding and the
      user who muted the finding.
    muteUpdateTime: Output only. The most recent time this finding was muted
      or unmuted.
    name: The [relative resource name](https://cloud.google.com/apis/design/re
      source_names#relative_resource_name) of the finding. The following list
      shows some examples: + `organizations/{organization_id}/sources/{source_
      id}/findings/{finding_id}` + `organizations/{organization_id}/sources/{s
      ource_id}/locations/{location_id}/findings/{finding_id}` +
      `folders/{folder_id}/sources/{source_id}/findings/{finding_id}` + `folde
      rs/{folder_id}/sources/{source_id}/locations/{location_id}/findings/{fin
      ding_id}` +
      `projects/{project_id}/sources/{source_id}/findings/{finding_id}` + `pro
      jects/{project_id}/sources/{source_id}/locations/{location_id}/findings/
      {finding_id}`
    nextSteps: Steps to address the finding.
    notebook: Notebook associated with the finding.
    orgPolicies: Contains information about the org policies associated with
      the finding.
    parent: The relative resource name of the source and location the finding
      belongs to. See: https://cloud.google.com/apis/design/resource_names#rel
      ative_resource_name This field is immutable after creation time. The
      following list shows some examples: +
      `organizations/{organization_id}/sources/{source_id}` +
      `folders/{folders_id}/sources/{source_id}` +
      `projects/{projects_id}/sources/{source_id}` + `organizations/{organizat
      ion_id}/sources/{source_id}/locations/{location_id}` +
      `folders/{folders_id}/sources/{source_id}/locations/{location_id}` +
      `projects/{projects_id}/sources/{source_id}/locations/{location_id}`
    parentDisplayName: Output only. The human readable display name of the
      finding source such as "Event Threat Detection" or "Security Health
      Analytics".
    processes: Represents operating system processes associated with the
      Finding.
    resourceName: Immutable. For findings on Google Cloud resources, the full
      resource name of the Google Cloud resource this finding is for. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
      When the finding is for a non-Google Cloud resource, the resourceName
      can be a customer or partner defined string.
    securityMarks: Output only. User specified security marks. These marks are
      entirely managed by the user and come from the SecurityMarks resource
      that belongs to the finding.
    securityPosture: The security posture associated with the finding.
    severity: The severity of the finding. This field is managed by the source
      that writes the finding.
    sourceProperties: Source specific properties. These properties are managed
      by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.
    state: Output only. The state of the finding.
    vulnerability: Represents vulnerability-specific fields like CVE and CVSS
      scores. CVE stands for Common Vulnerabilities and Exposures
      (https://cve.mitre.org/about/)
  """

    class FindingClassValueValuesEnum(_messages.Enum):
        """The class of the finding.

    Values:
      FINDING_CLASS_UNSPECIFIED: Unspecified finding class.
      THREAT: Describes unwanted or malicious activity.
      VULNERABILITY: Describes a potential weakness in software that increases
        risk to Confidentiality & Integrity & Availability.
      MISCONFIGURATION: Describes a potential weakness in cloud resource/asset
        configuration that increases risk.
      OBSERVATION: Describes a security observation that is for informational
        purposes.
      SCC_ERROR: Describes an error that prevents some SCC functionality.
      POSTURE_VIOLATION: Describes a potential security risk due to a change
        in the security posture.
    """
        FINDING_CLASS_UNSPECIFIED = 0
        THREAT = 1
        VULNERABILITY = 2
        MISCONFIGURATION = 3
        OBSERVATION = 4
        SCC_ERROR = 5
        POSTURE_VIOLATION = 6

    class MuteValueValuesEnum(_messages.Enum):
        """Indicates the mute state of a finding (either muted, unmuted or
    undefined). Unlike other attributes of a finding, a finding provider
    shouldn't set the value of mute.

    Values:
      MUTE_UNSPECIFIED: Unspecified.
      MUTED: Finding has been muted.
      UNMUTED: Finding has been unmuted.
      UNDEFINED: Finding has never been muted/unmuted.
    """
        MUTE_UNSPECIFIED = 0
        MUTED = 1
        UNMUTED = 2
        UNDEFINED = 3

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the finding. This field is managed by the source that
    writes the finding.

    Values:
      SEVERITY_UNSPECIFIED: This value is used for findings when a source
        doesn't write a severity value.
      CRITICAL: Vulnerability: A critical vulnerability is easily discoverable
        by an external actor, exploitable, and results in the direct ability
        to execute arbitrary code, exfiltrate data, and otherwise gain
        additional access and privileges to cloud resources and workloads.
        Examples include publicly accessible unprotected user data and public
        SSH access with weak or no passwords. Threat: Indicates a threat that
        is able to access, modify, or delete data or execute unauthorized code
        within existing resources.
      HIGH: Vulnerability: A high risk vulnerability can be easily discovered
        and exploited in combination with other vulnerabilities in order to
        gain direct access and the ability to execute arbitrary code,
        exfiltrate data, and otherwise gain additional access and privileges
        to cloud resources and workloads. An example is a database with weak
        or no passwords that is only accessible internally. This database
        could easily be compromised by an actor that had access to the
        internal network. Threat: Indicates a threat that is able to create
        new computational resources in an environment but not able to access
        data or execute code in existing resources.
      MEDIUM: Vulnerability: A medium risk vulnerability could be used by an
        actor to gain access to resources or privileges that enable them to
        eventually (through multiple steps or a complex exploit) gain access
        and the ability to execute arbitrary code or exfiltrate data. An
        example is a service account with access to more projects than it
        should have. If an actor gains access to the service account, they
        could potentially use that access to manipulate a project the service
        account was not intended to. Threat: Indicates a threat that is able
        to cause operational impact but may not access data or execute
        unauthorized code.
      LOW: Vulnerability: A low risk vulnerability hampers a security
        organization's ability to detect vulnerabilities or active threats in
        their deployment, or prevents the root cause investigation of security
        issues. An example is monitoring and logs being disabled for resource
        configurations and access. Threat: Indicates a threat that has
        obtained minimal access to an environment but is not able to access
        data, execute code, or create resources.
    """
        SEVERITY_UNSPECIFIED = 0
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the finding.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: The finding requires attention and has not been addressed yet.
      INACTIVE: The finding has been fixed, triaged as a non-issue or
        otherwise addressed and is no longer active.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ContactsValue(_messages.Message):
        """Output only. Map containing the points of contact for the given
    finding. The key represents the type of contact, while the value contains
    a list of all the contacts that pertain. Please refer to:
    https://cloud.google.com/resource-manager/docs/managing-notification-
    contacts#notification-categories { "security": { "contacts": [ { "email":
    "person1@company.com" }, { "email": "person2@company.com" } ] } }

    Messages:
      AdditionalProperty: An additional property for a ContactsValue object.

    Fields:
      additionalProperties: Additional properties of type ContactsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ContactsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudSecuritycenterV2ContactDetails attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudSecuritycenterV2ContactDetails', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExternalSystemsValue(_messages.Message):
        """Output only. Third party SIEM/SOAR fields within SCC, contains
    external system information and external system finding fields.

    Messages:
      AdditionalProperty: An additional property for a ExternalSystemsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExternalSystemsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExternalSystemsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudSecuritycenterV2ExternalSystem attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudSecuritycenterV2ExternalSystem', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SourcePropertiesValue(_messages.Message):
        """Source specific properties. These properties are managed by the source
    that writes the finding. The key names in the source_properties map must
    be between 1 and 255 characters, and must start with a letter and contain
    alphanumeric characters or underscores only.

    Messages:
      AdditionalProperty: An additional property for a SourcePropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SourcePropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SourcePropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    access = _messages.MessageField('GoogleCloudSecuritycenterV2Access', 1)
    application = _messages.MessageField('GoogleCloudSecuritycenterV2Application', 2)
    attackExposure = _messages.MessageField('GoogleCloudSecuritycenterV2AttackExposure', 3)
    backupDisasterRecovery = _messages.MessageField('GoogleCloudSecuritycenterV2BackupDisasterRecovery', 4)
    canonicalName = _messages.StringField(5)
    category = _messages.StringField(6)
    cloudDlpDataProfile = _messages.MessageField('GoogleCloudSecuritycenterV2CloudDlpDataProfile', 7)
    cloudDlpInspection = _messages.MessageField('GoogleCloudSecuritycenterV2CloudDlpInspection', 8)
    compliances = _messages.MessageField('GoogleCloudSecuritycenterV2Compliance', 9, repeated=True)
    connections = _messages.MessageField('GoogleCloudSecuritycenterV2Connection', 10, repeated=True)
    contacts = _messages.MessageField('ContactsValue', 11)
    containers = _messages.MessageField('GoogleCloudSecuritycenterV2Container', 12, repeated=True)
    createTime = _messages.StringField(13)
    database = _messages.MessageField('GoogleCloudSecuritycenterV2Database', 14)
    description = _messages.StringField(15)
    eventTime = _messages.StringField(16)
    exfiltration = _messages.MessageField('GoogleCloudSecuritycenterV2Exfiltration', 17)
    externalSystems = _messages.MessageField('ExternalSystemsValue', 18)
    externalUri = _messages.StringField(19)
    files = _messages.MessageField('GoogleCloudSecuritycenterV2File', 20, repeated=True)
    findingClass = _messages.EnumField('FindingClassValueValuesEnum', 21)
    iamBindings = _messages.MessageField('GoogleCloudSecuritycenterV2IamBinding', 22, repeated=True)
    indicator = _messages.MessageField('GoogleCloudSecuritycenterV2Indicator', 23)
    kernelRootkit = _messages.MessageField('GoogleCloudSecuritycenterV2KernelRootkit', 24)
    kubernetes = _messages.MessageField('GoogleCloudSecuritycenterV2Kubernetes', 25)
    loadBalancers = _messages.MessageField('GoogleCloudSecuritycenterV2LoadBalancer', 26, repeated=True)
    logEntries = _messages.MessageField('GoogleCloudSecuritycenterV2LogEntry', 27, repeated=True)
    mitreAttack = _messages.MessageField('GoogleCloudSecuritycenterV2MitreAttack', 28)
    moduleName = _messages.StringField(29)
    mute = _messages.EnumField('MuteValueValuesEnum', 30)
    muteInitiator = _messages.StringField(31)
    muteUpdateTime = _messages.StringField(32)
    name = _messages.StringField(33)
    nextSteps = _messages.StringField(34)
    notebook = _messages.MessageField('GoogleCloudSecuritycenterV2Notebook', 35)
    orgPolicies = _messages.MessageField('GoogleCloudSecuritycenterV2OrgPolicy', 36, repeated=True)
    parent = _messages.StringField(37)
    parentDisplayName = _messages.StringField(38)
    processes = _messages.MessageField('GoogleCloudSecuritycenterV2Process', 39, repeated=True)
    resourceName = _messages.StringField(40)
    securityMarks = _messages.MessageField('GoogleCloudSecuritycenterV2SecurityMarks', 41)
    securityPosture = _messages.MessageField('GoogleCloudSecuritycenterV2SecurityPosture', 42)
    severity = _messages.EnumField('SeverityValueValuesEnum', 43)
    sourceProperties = _messages.MessageField('SourcePropertiesValue', 44)
    state = _messages.EnumField('StateValueValuesEnum', 45)
    vulnerability = _messages.MessageField('GoogleCloudSecuritycenterV2Vulnerability', 46)