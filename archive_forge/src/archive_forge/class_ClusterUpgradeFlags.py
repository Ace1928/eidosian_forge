from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
class ClusterUpgradeFlags:
    """Add flags to the cluster upgrade command surface."""

    def __init__(self, parser: parser_arguments.ArgumentInterceptor):
        self._parser = parser

    @property
    def parser(self):
        return self._parser

    def AddShowLinkedClusterUpgrade(self):
        """Adds the --show-linked-cluster-upgrade flag."""
        self.parser.add_argument('--show-linked-cluster-upgrade', action='store_true', default=None, help="        Shows the cluster upgrade feature information for the current fleet as\n        well as information for all other fleets linked in the same rollout\n        sequence (provided that the caller has permission to view the upstream\n        and downstream fleets). This displays cluster upgrade information for\n        fleets in the current fleet's rollout sequence in order of furthest\n        upstream to downstream.\n\n        To view the cluster upgrade feature information for the rollout\n        sequence containing the current fleet, run:\n\n          $ {command} --show-linked-cluster-upgrade\n        ")

    def AddDefaultUpgradeSoakingFlag(self):
        """Adds the --default-upgrade-soaking flag."""
        self.parser.add_argument('--default-upgrade-soaking', type=arg_parsers.Duration(), help='        Configures the default soaking duration for each upgrade propagating\n        through the current fleet to become "COMPLETE". Soaking begins after\n        all clusters in the fleet are on the target version, or after 30 days\n        if all cluster upgrades are not complete. Once an upgrade state becomes\n        "COMPLETE", it will automatically be propagated to the downstream\n        fleet. Max is 30 days.\n\n        To configure Rollout Sequencing for a fleet, this attribute must be\n        set. To do this while specifying a default soaking duration of 7 days,\n        run:\n\n          $ {command} --default-upgrade-soaking=7d\n        ')

    def AddUpgradeSoakingOverrideFlags(self, with_destructive=False):
        if with_destructive:
            group = self.parser.add_mutually_exclusive_group()
            self._AddRemoveUpgradeSoakingOverridesFlag(group)
            self._AddUpgradeSoakingOverrideFlags(group)
        else:
            self._AddUpgradeSoakingOverrideFlags(self.parser)

    def _AddRemoveUpgradeSoakingOverridesFlag(self, group: parser_arguments.ArgumentInterceptor):
        """Adds the --remove-upgrade-soaking-overrides flag.

    Args:
      group: The group that should contain the flag.
    """
        group.add_argument('--remove-upgrade-soaking-overrides', action='store_true', default=None, help='        Removes soaking time overrides for all upgrades propagating through the\n        current fleet. Consequently, all upgrades will follow the soak time\n        configured by `--default-upgrade-soaking` until new overrides are\n        configured with `--add_upgrade_soaking_override` and\n        `--upgrade_selector`.\n\n        To remove all configured soaking time overrides, run:\n\n          $ {command} --remove-upgrade-soaking-overrides\n        ')

    def _AddUpgradeSoakingOverrideFlags(self, group: parser_arguments.ArgumentInterceptor):
        """Adds upgrade soaking override flags.

    Args:
      group: The group that should contain the upgrade soaking override flags.
    """
        group = group.add_group(help='        Upgrade soaking override.\n\n        Defines a specific soaking time override for a particular upgrade\n        propagating through the current fleet that supercedes the default\n        soaking duration configured by `--default-upgrade-soaking`.\n\n        To set an upgrade soaking override of 12 hours for the upgrade with\n        name, `k8s_control_plane`, and version, `1.23.1-gke.1000`, run:\n\n          $ {command}               --add-upgrade-soaking-override=12h               --upgrade-selector=name="k8s_control_plane",version="1.23.1-gke.1000"\n        ')
        self._AddAddUpgradeSoakingOverrideFlag(group)
        self._AddUpgradeSelectorFlag(group)

    def _AddAddUpgradeSoakingOverrideFlag(self, group: parser_arguments.ArgumentInterceptor):
        """Adds the --add-upgrade-soaking-override flag.

    Args:
      group: The group that should contain the flag.
    """
        group.add_argument('--add-upgrade-soaking-override', type=arg_parsers.Duration(), required=True, help='        Overrides the soaking time for a particular upgrade name and version\n        propagating through the current fleet. Set soaking to 0 days to bypass\n        soaking and fast-forward the upgrade to the downstream fleet.\n\n        See `$ gcloud topic datetimes` for information on duration formats.\n        ')

    def _AddUpgradeSelectorFlag(self, group: parser_arguments.ArgumentInterceptor):
        """Adds the --ugprade-selector flag.

    Args:
      group: The group that should contain the flag.
    """
        group.add_argument('--upgrade-selector', type=UpgradeSelector(), required=True, help='        Name and version of the upgrade to be overridden where version is a\n        full GKE version. Currently, name can be either `k8s_control_plane` or\n        `k8s_node`.\n        ')

    def AddUpstreamFleetFlags(self, with_destructive=False):
        """Adds upstream fleet flags."""
        if with_destructive:
            group = self.parser.add_mutually_exclusive_group()
            self._AddUpstreamFleetFlag(group)
            self._AddResetUpstreamFleetFlag(group)
        else:
            self._AddUpstreamFleetFlag(self.parser)

    def _AddUpstreamFleetFlag(self, group: parser_arguments.ArgumentInterceptor):
        """Adds the --upstream-fleet flag.

    Args:
      group: The group that should contain the flag.
    """
        group.add_argument('--upstream-fleet', type=str, help='        The upstream fleet. GKE will finish upgrades on the upstream fleet\n        before applying the same upgrades to the current fleet.\n\n        To configure the upstream fleet, run:\n\n        $ {command}             --upstream-fleet={upstream_fleet}\n        ')

    def _AddResetUpstreamFleetFlag(self, group: parser_arguments.ArgumentInterceptor):
        """Adds the --reset-upstream-fleet flag.

    Args:
      group: The group that should contain the flag.
    """
        group.add_argument('--reset-upstream-fleet', action='store_true', default=None, help='        Clears the relationship between the current fleet and its upstream\n        fleet in the rollout sequence.\n\n        To remove the link between the current fleet and its upstream fleet,\n        run:\n\n          $ {command} --reset-upstream-fleet\n        ')