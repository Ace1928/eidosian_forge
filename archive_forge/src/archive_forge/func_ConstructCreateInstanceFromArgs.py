from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@classmethod
def ConstructCreateInstanceFromArgs(cls, sql_messages, args, original=None, instance_ref=None, release_track=DEFAULT_RELEASE_TRACK):
    """Constructs Instance for create request from base instance and args."""
    ShowZoneDeprecationWarnings(args)
    instance_resource = cls._ConstructBaseInstanceFromArgs(sql_messages, args, original, instance_ref)
    instance_resource.region = reducers.Region(args.region, _GetZone(args), _GetSecondaryZone(args))
    instance_resource.databaseVersion = ParseDatabaseVersion(sql_messages, args.database_version)
    instance_resource.masterInstanceName = args.master_instance_name
    instance_resource.rootPassword = args.root_password
    if IsBetaOrNewer(release_track) and args.IsSpecified('source_ip_address'):
        on_premises_configuration = reducers.OnPremisesConfiguration(sql_messages, args.source_ip_address, args.source_port)
        instance_resource.onPremisesConfiguration = on_premises_configuration
        return instance_resource
    instance_resource.settings = cls._ConstructCreateSettingsFromArgs(sql_messages, args, original, release_track)
    if args.master_instance_name:
        replication = sql_messages.Settings.ReplicationTypeValueValuesEnum.ASYNCHRONOUS
        if args.replica_type == 'FAILOVER':
            instance_resource.replicaConfiguration = sql_messages.ReplicaConfiguration(kind='sql#demoteMasterMysqlReplicaConfiguration', failoverTarget=True)
        if args.cascadable_replica:
            if instance_resource.replicaConfiguration:
                instance_resource.replicaConfiguration.cascadableReplica = args.cascadable_replica
            else:
                instance_resource.replicaConfiguration = sql_messages.ReplicaConfiguration(kind='sql#replicaConfiguration', cascadableReplica=args.cascadable_replica)
    else:
        replication = sql_messages.Settings.ReplicationTypeValueValuesEnum.SYNCHRONOUS
    if not args.replication:
        instance_resource.settings.replicationType = replication
    if args.failover_replica_name:
        instance_resource.failoverReplica = sql_messages.DatabaseInstance.FailoverReplicaValue(name=args.failover_replica_name)
    if args.collation:
        instance_resource.settings.collation = args.collation
    if IsBetaOrNewer(release_track) and args.IsSpecified('master_username'):
        if not args.IsSpecified('master_instance_name'):
            raise exceptions.RequiredArgumentException('--master-instance-name', 'To create a read replica of an external master instance, [--master-instance-name] must be specified')
        if not (args.IsSpecified('master_password') or args.IsSpecified('prompt_for_master_password')):
            raise exceptions.RequiredArgumentException('--master-password', 'To create a read replica of an external master instance, [--master-password] or [--prompt-for-master-password] must be specified')
        if args.prompt_for_master_password:
            args.master_password = console_io.PromptPassword('Master Instance Password: ')
        instance_resource.replicaConfiguration = reducers.ReplicaConfiguration(sql_messages, args.master_username, args.master_password, args.master_dump_file_path, args.master_ca_certificate_path, args.client_certificate_path, args.client_key_path)
    is_primary = instance_resource.masterInstanceName is None
    key_name = _GetAndValidateCmekKeyName(args, is_primary)
    if key_name:
        config = sql_messages.DiskEncryptionConfiguration(kind='sql#diskEncryptionConfiguration', kmsKeyName=key_name)
        instance_resource.diskEncryptionConfiguration = config
    return instance_resource