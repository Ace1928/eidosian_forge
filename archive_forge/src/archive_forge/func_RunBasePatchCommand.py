from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def RunBasePatchCommand(args, release_track):
    """Updates settings of a Cloud SQL instance using the patch api method.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    release_track: base.ReleaseTrack, the release track that this was run under.

  Returns:
    A dict object representing the operations resource describing the patch
    operation if the patch was successful.
  Raises:
    CancelledError: The user chose not to continue.
  """
    if args.diff and (not args.IsSpecified('format')):
        args.format = 'diff(old, new)'
    client = common_api_util.SqlClient(common_api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    validate.ValidateInstanceName(args.instance)
    validate.ValidateInstanceLocation(args)
    instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    if args.IsSpecified('simulate_maintenance_event'):
        for key in args.GetSpecifiedArgsDict():
            if key == 'instance':
                continue
            if key == 'simulate_maintenance_event':
                continue
            if not args.GetFlagArgument(key).is_global:
                raise exceptions.ArgumentError('`--simulate-maintenance-event` cannot be specified with other arguments excluding gCloud wide flags')
    if args.IsSpecified('no_backup'):
        if args.IsSpecified('enable_bin_log'):
            raise exceptions.ArgumentError('`--enable-bin-log` cannot be specified when --no-backup is specified')
        elif args.IsSpecified('enable_point_in_time_recovery'):
            raise exceptions.ArgumentError('`--enable-point-in-time-recovery` cannot be specified when --no-backup is specified')
    if args.IsKnownAndSpecified('failover_dr_replica_name'):
        if args.IsKnownAndSpecified('clear_failover_dr_replica_name'):
            raise exceptions.ArgumentError('`--failover-dr-replica-name` cannot be specified when --clear-failover-dr-replica-name is specified')
    if args.authorized_networks:
        api_util.InstancesV1Beta4.PrintAndConfirmAuthorizedNetworksOverwrite()
    original_instance_resource = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=instance_ref.project, instance=instance_ref.instance))
    patch_instance = command_util.InstancesV1Beta4.ConstructPatchInstanceFromArgs(sql_messages, args, original=original_instance_resource, release_track=release_track)
    patch_instance.project = instance_ref.project
    patch_instance.name = instance_ref.instance
    cleared_fields = _GetConfirmedClearedFields(args, patch_instance, original_instance_resource)
    if args.maintenance_window_any:
        cleared_fields.append('settings.maintenanceWindow')
    if args.IsKnownAndSpecified('clear_failover_dr_replica_name'):
        cleared_fields.append('replicationCluster')
    with sql_client.IncludeFields(cleared_fields):
        result_operation = sql_client.instances.Patch(sql_messages.SqlInstancesPatchRequest(databaseInstance=patch_instance, project=instance_ref.project, instance=instance_ref.instance))
    operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Patching Cloud SQL instance')
    log.UpdatedResource(instance_ref)
    changed_instance_resource = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=instance_ref.project, instance=instance_ref.instance))
    return _Result(changed_instance_resource, original_instance_resource)