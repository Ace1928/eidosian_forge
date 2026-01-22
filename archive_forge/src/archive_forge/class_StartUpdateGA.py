from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import rolling_action
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StartUpdateGA(base.Command):
    """Start update instances of managed instance group."""

    @classmethod
    def Args(cls, parser):
        _AddArgs(parser=parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        update_instances_utils.ValidateCanaryVersionFlag('--canary-version', args.canary_version)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        request = self.CreateRequest(args, client, holder.resources)
        return client.MakeRequests([request])

    def CreateRequest(self, args, client, resources):
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        igm_ref = resource_arg.ResolveAsResource(args, resources, default_scope=default_scope, scope_lister=scope_lister)
        if igm_ref.Collection() not in ['compute.instanceGroupManagers', 'compute.regionInstanceGroupManagers']:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        update_policy_type = update_instances_utils.ParseUpdatePolicyType('--type', args.type, client.messages)
        max_surge = update_instances_utils.ParseFixedOrPercent('--max-surge', 'max-surge', args.max_surge, client.messages)
        max_unavailable = update_instances_utils.ParseFixedOrPercent('--max-unavailable', 'max-unavailable', args.max_unavailable, client.messages)
        igm_info = managed_instance_groups_utils.GetInstanceGroupManagerOrThrow(igm_ref, client)
        if TEMPLATE_NAME_KEY in args.version:
            args.template = args.version[TEMPLATE_NAME_KEY]
        versions = []
        versions.append(update_instances_utils.ParseVersion(args, '--version', args.version, resources, client.messages))
        if args.canary_version:
            if TEMPLATE_NAME_KEY in args.canary_version:
                args.template = args.canary_version[TEMPLATE_NAME_KEY]
            versions.append(update_instances_utils.ParseVersion(args, '--canary-version', args.canary_version, resources, client.messages))
        managed_instance_groups_utils.ValidateVersions(igm_info, versions, resources, args.force)
        igm_version_names = {version.instanceTemplate: version.name for version in igm_info.versions}
        for version in versions:
            if not version.name:
                version.name = igm_version_names.get(version.instanceTemplate)
        update_policy = client.messages.InstanceGroupManagerUpdatePolicy(maxSurge=max_surge, maxUnavailable=max_unavailable, type=update_policy_type)
        if args.IsSpecified('minimal_action'):
            update_policy.minimalAction = update_instances_utils.ParseInstanceActionFlag('--minimal-action', args.minimal_action, client.messages.InstanceGroupManagerUpdatePolicy.MinimalActionValueValuesEnum)
        if args.IsSpecified('most_disruptive_allowed_action'):
            update_policy.mostDisruptiveAllowedAction = update_instances_utils.ParseInstanceActionFlag('--most-disruptive-allowed-action', args.most_disruptive_allowed_action, client.messages.InstanceGroupManagerUpdatePolicy.MostDisruptiveAllowedActionValueValuesEnum)
        if hasattr(args, 'min_ready'):
            update_policy.minReadySec = args.min_ready
        if hasattr(args, 'replacement_method'):
            replacement_method = update_instances_utils.ParseReplacementMethod(args.replacement_method, client.messages)
            update_policy.replacementMethod = replacement_method
        rolling_action.ValidateAndFixUpdaterAgainstStateful(update_policy, igm_info, client, args)
        igm_resource = client.messages.InstanceGroupManager(instanceTemplate=None, updatePolicy=update_policy, versions=versions)
        if hasattr(igm_ref, 'zone'):
            service = client.apitools_client.instanceGroupManagers
            request = client.messages.ComputeInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=igm_resource, project=igm_ref.project, zone=igm_ref.zone)
        elif hasattr(igm_ref, 'region'):
            service = client.apitools_client.regionInstanceGroupManagers
            request = client.messages.ComputeRegionInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=igm_resource, project=igm_ref.project, region=igm_ref.region)
        return (service, 'Patch', request)