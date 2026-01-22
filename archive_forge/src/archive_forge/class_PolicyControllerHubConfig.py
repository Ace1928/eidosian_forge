from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerHubConfig(_messages.Message):
    """Configuration for Policy Controller

  Enums:
    InstallSpecValueValuesEnum: The install_spec represents the intended state
      specified by the latest request that mutated install_spec in the feature
      spec, not the lifecycle state of the feature observed by the Hub feature
      controller that is reported in the feature state.

  Messages:
    DeploymentConfigsValue: Map of deployment configs to deployments
      ("admission", "audit", "mutation').

  Fields:
    auditIntervalSeconds: Sets the interval for Policy Controller Audit Scans
      (in seconds). When set to 0, this disables audit functionality
      altogether.
    constraintViolationLimit: The maximum number of audit violations to be
      stored in a constraint. If not set, the internal default (currently 20)
      will be used.
    deploymentConfigs: Map of deployment configs to deployments ("admission",
      "audit", "mutation').
    exemptableNamespaces: The set of namespaces that are excluded from Policy
      Controller checks. Namespaces do not need to currently exist on the
      cluster.
    installSpec: The install_spec represents the intended state specified by
      the latest request that mutated install_spec in the feature spec, not
      the lifecycle state of the feature observed by the Hub feature
      controller that is reported in the feature state.
    logDeniesEnabled: Logs all denies and dry run failures.
    monitoring: Monitoring specifies the configuration of monitoring.
    mutationEnabled: Enables the ability to mutate resources using Policy
      Controller.
    policyContent: Specifies the desired policy content on the cluster
    referentialRulesEnabled: Enables the ability to use Constraint Templates
      that reference to objects other than the object currently being
      evaluated.
  """

    class InstallSpecValueValuesEnum(_messages.Enum):
        """The install_spec represents the intended state specified by the latest
    request that mutated install_spec in the feature spec, not the lifecycle
    state of the feature observed by the Hub feature controller that is
    reported in the feature state.

    Values:
      INSTALL_SPEC_UNSPECIFIED: Spec is unknown.
      INSTALL_SPEC_NOT_INSTALLED: Request to uninstall Policy Controller.
      INSTALL_SPEC_ENABLED: Request to install and enable Policy Controller.
      INSTALL_SPEC_SUSPENDED: Request to suspend Policy Controller i.e. its
        webhooks. If Policy Controller is not installed, it will be installed
        but suspended.
      INSTALL_SPEC_DETACHED: Request to stop all reconciliation actions by
        PoCo Hub controller. This is a breakglass mechanism to stop PoCo Hub
        from affecting cluster resources.
    """
        INSTALL_SPEC_UNSPECIFIED = 0
        INSTALL_SPEC_NOT_INSTALLED = 1
        INSTALL_SPEC_ENABLED = 2
        INSTALL_SPEC_SUSPENDED = 3
        INSTALL_SPEC_DETACHED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DeploymentConfigsValue(_messages.Message):
        """Map of deployment configs to deployments ("admission", "audit",
    "mutation').

    Messages:
      AdditionalProperty: An additional property for a DeploymentConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DeploymentConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DeploymentConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerPolicyControllerDeploymentConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyControllerPolicyControllerDeploymentConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    auditIntervalSeconds = _messages.IntegerField(1)
    constraintViolationLimit = _messages.IntegerField(2)
    deploymentConfigs = _messages.MessageField('DeploymentConfigsValue', 3)
    exemptableNamespaces = _messages.StringField(4, repeated=True)
    installSpec = _messages.EnumField('InstallSpecValueValuesEnum', 5)
    logDeniesEnabled = _messages.BooleanField(6)
    monitoring = _messages.MessageField('PolicyControllerMonitoringConfig', 7)
    mutationEnabled = _messages.BooleanField(8)
    policyContent = _messages.MessageField('PolicyControllerPolicyContentSpec', 9)
    referentialRulesEnabled = _messages.BooleanField(10)