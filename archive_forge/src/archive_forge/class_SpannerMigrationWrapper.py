from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
class SpannerMigrationWrapper(binary_operations.StreamingBinaryBackedOperation):
    """Binary operation wrapper for spanner-migration-tool commands."""

    def __init__(self, **kwargs):
        super(SpannerMigrationWrapper, self).__init__(binary='spanner-migration-tool', install_if_missing=True, **kwargs)

    def _ParseSchemaArgs(self, source, prefix=None, source_profile=None, target=None, target_profile=None, dry_run=False, log_level=None, **kwargs):
        """"Parse args for the schema command."""
        del kwargs
        exec_args = ['schema']
        if source:
            exec_args.extend(['--source', source])
        if prefix:
            exec_args.extend(['--prefix', prefix])
        if source_profile:
            exec_args.extend(['--source-profile', source_profile])
        if target:
            exec_args.extend(['--target', target])
        if target_profile:
            exec_args.extend(['--target-profile', target_profile])
        if dry_run:
            exec_args.append('--dry-run')
        if log_level:
            exec_args.extend(['--log-level', log_level])
        return exec_args

    def _ParseDataArgs(self, source, session, prefix=None, skip_foreign_keys=False, source_profile=None, target=None, target_profile=None, write_limit=None, dry_run=False, log_level=None, **kwargs):
        """"Parse args for the data command."""
        del kwargs
        exec_args = ['data']
        if source:
            exec_args.extend(['--source', source])
        if session:
            exec_args.extend(['--session', session])
        if prefix:
            exec_args.extend(['--prefix', prefix])
        if skip_foreign_keys:
            exec_args.append('--skip-foreign-keys')
        if source_profile:
            exec_args.extend(['--source-profile', source_profile])
        if target:
            exec_args.extend(['--target', target])
        if target_profile:
            exec_args.extend(['--target-profile', target_profile])
        if write_limit:
            exec_args.extend(['--write-limit', write_limit])
        if dry_run:
            exec_args.append('--dry-run')
        if log_level:
            exec_args.extend(['--log-level', log_level])
        return exec_args

    def _ParseSchemaAndDataArgs(self, source, prefix=None, skip_foreign_keys=False, source_profile=None, target=None, target_profile=None, write_limit=None, dry_run=False, log_level=None, **kwargs):
        """"Parse args for the schema-and-data command."""
        del kwargs
        exec_args = ['schema-and-data']
        if source:
            exec_args.extend(['--source', source])
        if prefix:
            exec_args.extend(['--prefix', prefix])
        if skip_foreign_keys:
            exec_args.append('--skip-foreign-keys')
        if source_profile:
            exec_args.extend(['--source-profile', source_profile])
        if target:
            exec_args.extend(['--target', target])
        if target_profile:
            exec_args.extend(['--target-profile', target_profile])
        if write_limit:
            exec_args.extend(['--write-limit', write_limit])
        if dry_run:
            exec_args.append('--dry-run')
        if log_level:
            exec_args.extend(['--log-level', log_level])
        return exec_args

    def _ParseWebArgs(self, open_flag=False, port=None, **kwargs):
        """Parse args for the web command."""
        del kwargs
        exec_args = ['web']
        if open_flag:
            exec_args.append('--open')
        if port:
            exec_args.extend(['--port', port])
        return exec_args

    def ParseCleanupArgs(self, job_id, data_shard_ids=None, target_profile=None, datastream=False, dataflow=False, pub_sub=False, monitoring=False, log_level=None, **kwargs):
        """"Parse args for the cleanup command."""
        del kwargs
        exec_args = ['cleanup']
        if job_id:
            exec_args.extend(['--jobId', job_id])
        if data_shard_ids:
            exec_args.extend(['--dataShardIds', data_shard_ids])
        if target_profile:
            exec_args.extend(['--target-profile', target_profile])
        if datastream:
            exec_args.append('--datastream')
        if dataflow:
            exec_args.append('--dataflow')
        if pub_sub:
            exec_args.append('--pubsub')
        if monitoring:
            exec_args.append('--monitoring')
        if log_level:
            exec_args.append('--log-level')
        return exec_args

    def _ParseArgsForCommand(self, command, **kwargs):
        """Call the parser corresponding to the command."""
        if command == 'schema':
            return self._ParseSchemaArgs(**kwargs)
        elif command == 'data':
            return self._ParseDataArgs(**kwargs)
        elif command == 'schema-and-data':
            return self._ParseSchemaAndDataArgs(**kwargs)
        elif command == 'web':
            return self._ParseWebArgs(**kwargs)
        elif command == 'cleanup':
            return self.ParseCleanupArgs(**kwargs)
        else:
            raise binary_operations.InvalidOperationForBinary('Invalid Operation [{}] for spanner-migration-tool'.format(command))