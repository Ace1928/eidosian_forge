from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _GetUpdatedStream(self, stream, release_track, args):
    """Returns updated stream."""
    update_fields = []
    user_update_mask = args.update_mask or ''
    user_update_mask_list = user_update_mask.split(',')
    if release_track == base.ReleaseTrack.BETA:
        user_update_mask_list = util.UpdateV1alpha1ToV1MaskFields(user_update_mask_list)
    update_fields.extend(user_update_mask_list)
    if args.IsSpecified('display_name'):
        stream.displayName = args.display_name
    if release_track == base.ReleaseTrack.BETA:
        source_connection_profile_ref = args.CONCEPTS.source_name.Parse()
        source_field_name = 'source_name'
    else:
        source_connection_profile_ref = args.CONCEPTS.source.Parse()
        source_field_name = 'source'
    if args.IsSpecified(source_field_name):
        stream.sourceConfig.sourceConnectionProfile = source_connection_profile_ref.RelativeName()
        if source_field_name in update_fields:
            update_fields.remove(source_field_name)
            update_fields.append('source_config.source_connection_profile')
    if args.IsSpecified('oracle_source_config'):
        stream.sourceConfig.oracleSourceConfig = self._ParseOracleSourceConfig(args.oracle_source_config, release_track)
        update_fields = self._UpdateListWithFieldNamePrefixes(update_fields, 'oracle_source_config', 'source_config.')
    elif args.IsSpecified('mysql_source_config'):
        stream.sourceConfig.mysqlSourceConfig = self._ParseMysqlSourceConfig(args.mysql_source_config, release_track)
        update_fields = self._UpdateListWithFieldNamePrefixes(update_fields, 'mysql_source_config', 'source_config.')
    elif args.IsSpecified('postgresql_source_config'):
        stream.sourceConfig.postgresqlSourceConfig = self._ParsePostgresqlSourceConfig(args.postgresql_source_config)
        update_fields = self._UpdateListWithFieldNamePrefixes(update_fields, 'postgresql_source_config', 'source_config.')
    if release_track == base.ReleaseTrack.BETA:
        destination_connection_profile_ref = args.CONCEPTS.destination_name.Parse()
        destination_field_name = 'destination_name'
    else:
        destination_connection_profile_ref = args.CONCEPTS.destination.Parse()
        destination_field_name = 'destination'
    if args.IsSpecified(destination_field_name):
        stream.destinationConfig.destinationConnectionProfile = destination_connection_profile_ref.RelativeName()
        if destination_field_name in update_fields:
            update_fields.remove(destination_field_name)
            update_fields.append('destination_config.destination_connection_profile')
    if args.IsSpecified('gcs_destination_config'):
        stream.destinationConfig.gcsDestinationConfig = self._ParseGcsDestinationConfig(args.gcs_destination_config, release_track)
        update_fields = self._UpdateListWithFieldNamePrefixes(update_fields, 'gcs_destination_config', 'destination_config.')
    elif args.IsSpecified('bigquery_destination_config'):
        stream.destinationConfig.bigqueryDestinationConfig = self._ParseBigqueryDestinationConfig(args.bigquery_destination_config)
        update_fields = self._UpdateListWithFieldNamePrefixes(update_fields, 'bigquery_destination_config', 'destination_config.')
    if args.IsSpecified('backfill_none'):
        stream.backfillNone = self._messages.BackfillNoneStrategy()
        try:
            stream.reset('backfillAll')
        except AttributeError:
            pass
    elif args.IsSpecified('backfill_all'):
        backfill_all_strategy = self._GetBackfillAllStrategy(release_track, args)
        stream.backfillAll = backfill_all_strategy
        try:
            stream.reset('backfillNone')
        except AttributeError:
            pass
    if args.IsSpecified('state'):
        stream.state = self._messages.Stream.StateValueValuesEnum(args.state.upper())
    self._UpdateLabels(stream, args)
    return (stream, update_fields)