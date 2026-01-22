from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddDatabaseParamsFlags(parser, require_password=True):
    """Adds the database connectivity flags to the given parser."""
    database_params_group = parser.add_group(required=False, mutex=False)
    AddUsernameFlag(database_params_group, required=True)
    AddPasswordFlagGroup(database_params_group, required=require_password)
    AddHostFlag(database_params_group, required=True)
    AddPortFlag(database_params_group, required=True)