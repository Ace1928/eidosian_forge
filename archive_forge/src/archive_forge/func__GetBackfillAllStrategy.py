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
def _GetBackfillAllStrategy(self, release_track, args):
    """Gets BackfillAllStrategy message based on Stream objects source type."""
    if args.oracle_excluded_objects:
        return self._messages.BackfillAllStrategy(oracleExcludedObjects=util.ParseOracleRdbmsFile(self._messages, args.oracle_excluded_objects, release_track))
    elif args.mysql_excluded_objects:
        return self._messages.BackfillAllStrategy(mysqlExcludedObjects=util.ParseMysqlRdbmsFile(self._messages, args.mysql_excluded_objects, release_track))
    elif args.postgresql_excluded_objects:
        return self._messages.BackfillAllStrategy(postgresqlExcludedObjects=util.ParsePostgresqlRdbmsFile(self._messages, args.postgresql_excluded_objects))
    return self._messages.BackfillAllStrategy()