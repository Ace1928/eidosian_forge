from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
class PocoFlagParser:
    """Converts PocoFlag arguments to internal representations.

  hub_cfg references the PolicyControllerHubConfig object in:
  third_party/py/googlecloudsdk/generated_clients/apis/gkehub/v1alpha/gkehub_v1alpha_messages.py
  """

    def __init__(self, args: parser_extensions.Namespace, msgs):
        self.args = args
        self.messages = msgs

    def update_audit_interval(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.audit_interval:
            hub_cfg.auditIntervalSeconds = self.args.audit_interval
        return hub_cfg

    def update_constraint_violation_limit(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.constraint_violation_limit:
            hub_cfg.constraintViolationLimit = self.args.constraint_violation_limit
        return hub_cfg

    def update_exemptable_namespaces(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.clear_exemptable_namespaces:
            namespaces = []
            hub_cfg.exemptableNamespaces = namespaces
        if self.args.exemptable_namespaces:
            namespaces = self.args.exemptable_namespaces.split(',')
            hub_cfg.exemptableNamespaces = namespaces
        return hub_cfg

    def update_log_denies(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.log_denies:
            hub_cfg.logDeniesEnabled = True
        if self.args.no_log_denies:
            hub_cfg.logDeniesEnabled = False
        return hub_cfg

    def update_mutation(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.mutation:
            hub_cfg.mutationEnabled = True
        if self.args.no_mutation:
            hub_cfg.mutationEnabled = False
        return hub_cfg

    def update_referential_rules(self, hub_cfg: messages.Message) -> messages.Message:
        if self.args.referential_rules:
            hub_cfg.referentialRulesEnabled = True
        if self.args.no_referential_rules:
            hub_cfg.referentialRulesEnabled = False
        return hub_cfg

    @property
    def monitoring_backend_cfg(self) -> messages.Message:
        return self.messages.PolicyControllerMonitoringConfig

    @property
    def monitoring_backend_enum(self) -> messages.Message:
        return self.monitoring_backend_cfg.BackendsValueListEntryValuesEnum

    def _get_monitoring_enum(self, backend) -> messages.Message:
        internal_name = constants.MONITORING_BACKENDS.get(backend)
        if internal_name is None or not hasattr(self.monitoring_backend_enum, constants.MONITORING_BACKENDS[backend]):
            raise exceptions.InvalidMonitoringBackendError('No such monitoring backend: {}'.format(backend))
        else:
            return getattr(self.monitoring_backend_enum, constants.MONITORING_BACKENDS[backend])

    def update_monitoring(self, hub_cfg: messages.Message) -> messages.Message:
        """Sets or removes monitoring backends based on args."""
        if self.args.no_monitoring:
            config = self.messages.PolicyControllerMonitoringConfig(backends=[])
            hub_cfg.monitoring = config
        if self.args.monitoring:
            backends = [self._get_monitoring_enum(backend) for backend in self.args.monitoring.split(',')]
            config = self.messages.PolicyControllerMonitoringConfig(backends=backends)
            hub_cfg.monitoring = config
        return hub_cfg

    @property
    def bundle_message(self) -> messages.Message:
        """Returns an reference to the BundlesValue class for this API channel."""
        return self.messages.PolicyControllerPolicyContentSpec.BundlesValue

    def update_default_bundles(self, hub_cfg: messages.Message) -> messages.Message:
        """Sets default bundles based on args.

    This function assumes that the hub config is being initialized for the first
    time.

    Args:
      hub_cfg: A 'PolicyControllerHubConfig' proto message.

    Returns:
      A modified hub_config, adding the default bundle; or unmodified if the
      --no-default-bundles flag is specified.
    """
        if self.args.no_default_bundles:
            return hub_cfg
        policy_content_spec = self._get_policy_content(hub_cfg)
        bundles = protos.additional_properties_to_dict(policy_content_spec.bundles)
        bundles[DEFAULT_BUNDLE_NAME] = self.messages.PolicyControllerBundleInstallSpec()
        policy_content_spec.bundles = protos.set_additional_properties(self.bundle_message(), bundles)
        hub_cfg.policyContent = policy_content_spec
        return hub_cfg

    def is_feature_update(self) -> bool:
        return self.args.fleet_default_member_config or self.args.no_fleet_default_member_config

    def load_fleet_default_cfg(self) -> messages.Message:
        if self.args.fleet_default_member_config:
            config_path = path.expanduser(self.args.fleet_default_member_config)
            data = console_io.ReadFromFileOrStdin(config_path, binary=False)
            return util.Import(self.messages.PolicyControllerMembershipSpec, data)

    @property
    def template_lib_cfg(self) -> messages.Message:
        return self.messages.PolicyControllerTemplateLibraryConfig

    @property
    def template_lib_enum(self) -> messages.Message:
        return self.template_lib_cfg.InstallationValueValuesEnum

    def _get_policy_content(self, poco_cfg: messages.Message) -> messages.Message:
        """Get or create new PolicyControllerPolicyContentSpec."""
        if poco_cfg.policyContent is None:
            return self.messages.PolicyControllerPolicyContentSpec()
        return poco_cfg.policyContent

    def update_version(self, poco: messages.Message) -> messages.Message:
        if self.args.version:
            poco.version = self.args.version
        return poco

    def use_default_cfg(self) -> bool:
        return self.args.origin and self.args.origin == 'FLEET'

    def set_default_cfg(self, feature: messages.Message, membership: messages.Message) -> messages.Message:
        """Sets membership to the default fleet configuration.

    Args:
      feature: The feature spec for the project.
      membership: The membership spec being updated.

    Returns:
      Updated MembershipFeatureSpec.
    Raises:
      MissingFleetDefaultMemberConfig: If none exists on the feature.
    """
        if feature.fleetDefaultMemberConfig is None:
            project = feature.name.split('/')[1]
            msg = "No fleet default member config specified for project {}. Use '... enable --fleet-default-member-config=config.yaml'."
            raise exceptions.MissingFleetDefaultMemberConfig(msg.format(project))
        self.set_origin_fleet(membership)
        membership.policycontroller = feature.fleetDefaultMemberConfig.policycontroller

    def set_origin_fleet(self, membership: messages.Message) -> messages.Message:
        membership.origin = self.messages.Origin(type=self.messages.Origin.TypeValueValuesEnum.FLEET)