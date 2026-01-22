import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GetConfigurationChanges(args, release_track=base.ReleaseTrack.GA):
    """Returns a list of changes shared by multiple resources, based on the flags set."""
    changes = []
    if hasattr(args, 'image') and args.image is not None:
        changes.append(config_changes.ImageChange(args.image))
    if _HasEnvChanges(args):
        changes.append(_GetEnvChanges(args))
    if _HasCloudSQLChanges(args):
        region = GetRegion(args)
        project = getattr(args, 'project', None) or properties.VALUES.core.project.Get(required=True)
        if _EnabledCloudSqlApiRequired(args):
            _CheckCloudSQLApiEnablement()
        changes.append(config_changes.CloudSQLChanges.FromArgs(project=project, region=region, args=args))
    if FlagIsExplicitlySet(args, 'remove_volume_mount') and args.remove_volume_mount or (FlagIsExplicitlySet(args, 'clear_volume_mounts') and args.clear_volume_mounts):
        changes.append(config_changes.RemoveVolumeMountChange(removed_mounts=args.remove_volume_mount, clear_mounts=args.clear_volume_mounts))
    if FlagIsExplicitlySet(args, 'remove_volume') and args.remove_volume or (FlagIsExplicitlySet(args, 'clear_volumes') and args.clear_volumes):
        changes.append(config_changes.RemoveVolumeChange(args.remove_volume, args.clear_volumes))
    if _HasSecretsChanges(args):
        changes.extend(_GetSecretsChanges(args))
    if FlagIsExplicitlySet(args, 'add_volume') and args.add_volume:
        changes.append(config_changes.AddVolumeChange(args.add_volume, release_track))
    if FlagIsExplicitlySet(args, 'add_volume_mount') and args.add_volume_mount:
        changes.append(config_changes.AddVolumeMountChange(new_mounts=args.add_volume_mount))
    if _HasConfigMapsChanges(args):
        changes.extend(_GetConfigMapsChanges(args))
    if 'cpu' in args and args.cpu:
        changes.append(config_changes.ResourceChanges(cpu=args.cpu))
    if 'memory' in args and args.memory:
        changes.append(config_changes.ResourceChanges(memory=args.memory))
    if 'gpu' in args and args.gpu:
        changes.append(config_changes.ResourceChanges(gpu=args.gpu))
    if 'service_account' in args and args.service_account:
        changes.append(config_changes.ServiceAccountChanges(service_account=args.service_account))
    if _HasLabelChanges(args):
        additions = args.labels if FlagIsExplicitlySet(args, 'labels') else args.update_labels
        diff = labels_util.Diff(additions=additions, subtractions=args.remove_labels if 'remove_labels' in args else [], clear=args.clear_labels if 'clear_labels' in args else False)
        if diff.MayHaveUpdates():
            changes.append(config_changes.LabelChanges(diff))
    if 'vpc_connector' in args and args.vpc_connector:
        changes.append(config_changes.VpcConnectorChange(args.vpc_connector))
    if FlagIsExplicitlySet(args, 'vpc_egress'):
        changes.append(config_changes.SetTemplateAnnotationChange(container_resource.EGRESS_SETTINGS_ANNOTATION, args.vpc_egress))
    if 'clear_vpc_connector' in args and args.clear_vpc_connector:
        changes.append(config_changes.ClearVpcConnectorChange())
    if 'command' in args and args.command is not None:
        changes.append(config_changes.ContainerCommandChange(args.command))
    if 'args' in args and args.args is not None:
        changes.append(config_changes.ContainerArgsChange(args.args))
    if FlagIsExplicitlySet(args, 'binary_authorization'):
        changes.append(config_changes.SetAnnotationChange(k8s_object.BINAUTHZ_POLICY_ANNOTATION, args.binary_authorization))
    if FlagIsExplicitlySet(args, 'clear_binary_authorization'):
        changes.append(config_changes.DeleteAnnotationChange(k8s_object.BINAUTHZ_POLICY_ANNOTATION))
    if FlagIsExplicitlySet(args, 'breakglass'):
        changes.append(config_changes.SetAnnotationChange(k8s_object.BINAUTHZ_BREAKGLASS_ANNOTATION, args.breakglass))
    if FlagIsExplicitlySet(args, 'key'):
        changes.append(config_changes.SetTemplateAnnotationChange(container_resource.CMEK_KEY_ANNOTATION, args.key))
    if FlagIsExplicitlySet(args, 'post_key_revocation_action_type'):
        changes.append(config_changes.SetTemplateAnnotationChange(container_resource.POST_CMEK_KEY_REVOCATION_ACTION_TYPE_ANNOTATION, args.post_key_revocation_action_type))
    if FlagIsExplicitlySet(args, 'encryption_key_shutdown_hours'):
        changes.append(config_changes.SetTemplateAnnotationChange(container_resource.ENCRYPTION_KEY_SHUTDOWN_HOURS_ANNOTATION, args.encryption_key_shutdown_hours))
    if FlagIsExplicitlySet(args, 'clear_key'):
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.CMEK_KEY_ANNOTATION))
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.POST_CMEK_KEY_REVOCATION_ACTION_TYPE_ANNOTATION))
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.ENCRYPTION_KEY_SHUTDOWN_HOURS_ANNOTATION))
    if FlagIsExplicitlySet(args, 'clear_post_key_revocation_action_type'):
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.POST_CMEK_KEY_REVOCATION_ACTION_TYPE_ANNOTATION))
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.ENCRYPTION_KEY_SHUTDOWN_HOURS_ANNOTATION))
    if FlagIsExplicitlySet(args, 'clear_encryption_key_shutdown_hours'):
        changes.append(config_changes.DeleteTemplateAnnotationChange(container_resource.ENCRYPTION_KEY_SHUTDOWN_HOURS_ANNOTATION))
    if FlagIsExplicitlySet(args, 'description'):
        changes.append(config_changes.SetAnnotationChange(k8s_object.DESCRIPTION_ANNOTATION, args.description))
    if 'execution_environment' in args and args.execution_environment:
        changes.append(config_changes.SandboxChange(args.execution_environment))
    if FlagIsExplicitlySet(args, 'network') or FlagIsExplicitlySet(args, 'subnet') or FlagIsExplicitlySet(args, 'network_tags') or FlagIsExplicitlySet(args, 'clear_network_tags'):
        network_tags_is_set = FlagIsExplicitlySet(args, 'clear_network_tags')
        network_tags = None
        if FlagIsExplicitlySet(args, 'network_tags'):
            network_tags_is_set = True
            network_tags = args.network_tags
        changes.append(config_changes.NetworkInterfacesChange(FlagIsExplicitlySet(args, 'network'), args.network, FlagIsExplicitlySet(args, 'subnet'), args.subnet, network_tags_is_set, network_tags))
    if 'clear_network' in args and args.clear_network:
        changes.append(config_changes.ClearNetworkInterfacesChange())
    if _HasCustomAudiencesChanges(args):
        changes.append(config_changes.CustomAudiencesChanges(args))
    if FlagIsExplicitlySet(args, 'remove_containers'):
        changes.append(config_changes.RemoveContainersChange.FromContainerNames(args.remove_containers))
        changes.append(config_changes.ContainerDependenciesChange())
    if FlagIsExplicitlySet(args, 'containers'):
        for container_name, container_args in args.containers.items():
            changes.extend(_GetContainerConfigurationChanges(container_args, container_name=container_name))
    if FlagIsExplicitlySet(args, 'mesh'):
        if args.mesh:
            changes.append(config_changes.SetTemplateAnnotationChange(revision.MESH_ANNOTATION, args.mesh))
        else:
            changes.append(config_changes.DeleteTemplateAnnotationChange(revision.MESH_ANNOTATION))
    if FlagIsExplicitlySet(args, 'base_image'):
        changes.append(config_changes.IngressContainerBaseImagesAnnotationChange(base_image=args.base_image))
    if FlagIsExplicitlySet(args, 'clear_base_image'):
        changes.append(config_changes.IngressContainerBaseImagesAnnotationChange(base_image=None))
    return changes