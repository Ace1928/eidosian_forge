from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
class DescribeCommand(feature_base.FeatureCommand, ClusterUpgradeCommand):
    """Command for describing a Scope's Cluster Upgrade Feature."""

    @staticmethod
    def GetProjectFromScopeName(name):
        """Extracts the project name from the full Scope resource name."""
        return name.split('/')[1]

    @staticmethod
    def FormatDurations(cluster_upgrade_spec):
        """Formats display strings for all cluster upgrade duration fields."""
        if cluster_upgrade_spec.postConditions is not None:
            default_soaking = cluster_upgrade_spec.postConditions.soaking
            if default_soaking is not None:
                cluster_upgrade_spec.postConditions.soaking = DescribeCommand.DisplayDuration(default_soaking)
        for override in cluster_upgrade_spec.gkeUpgradeOverrides:
            if override.postConditions is not None:
                override_soaking = override.postConditions.soaking
                if override_soaking is not None:
                    override.postConditions.soaking = DescribeCommand.DisplayDuration(override_soaking)
        return cluster_upgrade_spec

    @staticmethod
    def DisplayDuration(proto_duration_string):
        """Returns the display string for a duration value."""
        duration = times.ParseDuration(proto_duration_string)
        iso_duration = times.FormatDuration(duration)
        return re.sub('[-PT]', '', iso_duration).lower()

    def GetScopeWithClusterUpgradeInfo(self, scope, feature):
        """Adds Cluster Upgrade Feature information to describe Scope response."""
        scope_name = ClusterUpgradeCommand.GetScopeNameWithProjectNumber(scope.name)
        if self.args.IsKnownAndSpecified('show_cluster_upgrade') and self.args.show_cluster_upgrade:
            return self.AddClusterUpgradeInfoToScope(scope, scope_name, feature)
        elif self.args.IsKnownAndSpecified('show_linked_cluster_upgrade') and self.args.show_linked_cluster_upgrade:
            serialized_scope = resource_projector.MakeSerializable(scope)
            serialized_scope['clusterUpgrades'] = self.GetLinkedClusterUpgradeScopes(scope_name, feature)
            return serialized_scope
        return scope

    def AddClusterUpgradeInfoToScope(self, scope, scope_name, feature):
        serialized_scope = resource_projector.MakeSerializable(scope)
        serialized_scope['clusterUpgrade'] = self.GetClusterUpgradeInfoForScope(scope_name, feature)
        return serialized_scope

    def GetClusterUpgradeInfoForScope(self, scope_name, feature):
        """Gets Cluster Upgrade Feature information for the provided Scope."""
        scope_specs = self.hubclient.ToPyDict(feature.scopeSpecs)
        if scope_name not in scope_specs or not scope_specs[scope_name].clusterupgrade:
            msg = 'Cluster Upgrade feature is not configured for Scope: {}.'.format(scope_name)
            raise exceptions.Error(msg)
        state = self.hubclient.ToPyDefaultDict(self.messages.ScopeFeatureState, feature.scopeStates)[scope_name].clusterupgrade or self.messages.ClusterUpgradeScopeState()
        return {'scope': scope_name, 'state': state, 'spec': DescribeCommand.FormatDurations(scope_specs[scope_name].clusterupgrade)}

    def GetLinkedClusterUpgradeScopes(self, scope_name, feature):
        """Gets Cluster Upgrade Feature information for the entire sequence."""
        current_project = DescribeCommand.GetProjectFromScopeName(scope_name)
        visited = set([scope_name])

        def UpTheStream(cluster_upgrade):
            """Recursively gets information for the upstream Scopes."""
            upstream_spec = cluster_upgrade.get('spec', None)
            upstream_scopes = upstream_spec.upstreamScopes if upstream_spec else None
            if not upstream_scopes:
                return [cluster_upgrade]
            upstream_scope_name = upstream_scopes[0]
            if upstream_scope_name in visited:
                return [cluster_upgrade]
            visited.add(upstream_scope_name)
            upstream_scope_project = DescribeCommand.GetProjectFromScopeName(upstream_scope_name)
            upstream_feature = feature if upstream_scope_project == current_project else self.GetFeature(project=upstream_scope_project)
            try:
                upstream_cluster_upgrade = self.GetClusterUpgradeInfoForScope(upstream_scope_name, upstream_feature)
            except exceptions.Error as e:
                log.warning(e)
                return [cluster_upgrade]
            return UpTheStream(upstream_cluster_upgrade) + [cluster_upgrade]

        def DownTheStream(cluster_upgrade):
            """Recursively gets information for the downstream Scopes."""
            downstream_state = cluster_upgrade.get('state', None)
            downstream_scopes = downstream_state.downstreamScopes if downstream_state else None
            if not downstream_scopes:
                return [cluster_upgrade]
            downstream_scope_name = downstream_scopes[0]
            if downstream_scope_name in visited:
                return [cluster_upgrade]
            visited.add(downstream_scope_name)
            downstream_scope_project = DescribeCommand.GetProjectFromScopeName(downstream_scope_name)
            downstream_feature = feature if downstream_scope_project == current_project else self.GetFeature(project=downstream_scope_project)
            downstream_cluster_upgrade = self.GetClusterUpgradeInfoForScope(downstream_scope_name, downstream_feature)
            return [cluster_upgrade] + DownTheStream(downstream_cluster_upgrade)
        current_cluster_upgrade = self.GetClusterUpgradeInfoForScope(scope_name, feature)
        upstream_cluster_upgrades = UpTheStream(current_cluster_upgrade)[:-1]
        downstream_cluster_upgrades = DownTheStream(current_cluster_upgrade)[1:]
        return upstream_cluster_upgrades + [current_cluster_upgrade] + downstream_cluster_upgrades