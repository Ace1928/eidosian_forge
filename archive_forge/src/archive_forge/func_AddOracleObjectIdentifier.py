from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddOracleObjectIdentifier(parser):
    """Adds a --oracle-schema & --oracle-table flags to the given parser."""
    oracle_object_parser = parser.add_group()
    oracle_object_parser.add_argument('--oracle-schema', help='Oracle schema for the object.', required=True)
    oracle_object_parser.add_argument('--oracle-table', help='Oracle table for the object.', required=True)