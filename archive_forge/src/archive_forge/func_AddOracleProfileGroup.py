from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddOracleProfileGroup(parser, required=True):
    """Adds necessary oracle profile flags to the given parser."""
    oracle_profile = parser.add_group()
    oracle_profile.add_argument('--oracle-hostname', help='IP or hostname of the oracle source database.', required=required)
    oracle_profile.add_argument('--oracle-port', help='Network port of the oracle source database.', required=required, type=int)
    oracle_profile.add_argument('--oracle-username', help='Username Datastream will use to connect to the database.', required=required)
    oracle_profile.add_argument('--database-service', help='Database service for the Oracle connection.', required=required)
    password_group = oracle_profile.add_group(required=required, mutex=True)
    password_group.add_argument('--oracle-password', help='          Password for the user that Datastream will be using to\n          connect to the database.\n          This field is not returned on request, and the value is encrypted\n          when stored in Datastream.')
    password_group.add_argument('--oracle-prompt-for-password', action='store_true', help='Prompt for the password used to connect to the database.')