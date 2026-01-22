from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddBackfillStrategyGroup(parser, required=True):
    """Adds a --backfiill-all or --backfill-none flag to the given parser."""
    backfill_group = parser.add_group(required=required, mutex=True)
    backfill_group.add_argument('--backfill-none', help='Do not automatically backfill any objects. This flag is equivalent\n      to selecting the Manual backfill type in the Google Cloud console.', action='store_true')
    backfill_all_group = backfill_group.add_group()
    backfill_all_group.add_argument('--backfill-all', help='Automatically backfill objects included in the stream source\n      configuration. Specific objects can be excluded. This flag is equivalent\n      to selecting the Automatic backfill type in the Google Cloud console.', action='store_true')
    backfill_all_excluded_objects = backfill_all_group.add_group(mutex=True)
    backfill_all_excluded_objects.add_argument('--oracle-excluded-objects', help=_ORACLE_EXCLUDED_OBJECTS_HELP_TEXT)
    backfill_all_excluded_objects.add_argument('--mysql-excluded-objects', help=_MYSQL_EXCLUDED_OBJECTS_HELP_TEXT)
    backfill_all_excluded_objects.add_argument('--postgresql-excluded-objects', help=_POSTGRESQL_EXCLUDED_OBJECTS_HELP_TEXT)