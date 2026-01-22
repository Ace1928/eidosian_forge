from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def DatabaseFlags(sql_messages, settings=None, database_flags=None, clear_database_flags=False):
    """Generates the database flags for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    settings: sql_messages.Settings, the original settings, if the previous
      state is needed.
    database_flags: dict of flags.
    clear_database_flags: boolean, True if flags should be cleared.

  Returns:
    list of sql_messages.DatabaseFlags objects
  """
    updated_flags = []
    if database_flags:
        for name, value in sorted(database_flags.items()):
            updated_flags.append(sql_messages.DatabaseFlags(name=name, value=value))
    elif clear_database_flags:
        updated_flags = []
    elif settings:
        updated_flags = settings.databaseFlags
    return updated_flags