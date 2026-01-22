from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateHiveDatabaseName(db_name):
    """Validates the hive database name.

  Args:
    db_name: the hive database name.

  Returns:
    the hive database name.
  Raises:
    BadArgumentException: when the database name doesn't conform to the pattern
    or is longer than 64 characters.
  """
    pattern = re.compile('^[0-9a-zA-Z$_-]+$')
    if not pattern.match(db_name):
        raise exceptions.BadArgumentException('--hive-database-name', 'hive database name must start with an alphanumeric character, and contain only the following characters: letters, numbers, dashes (-), and underscores (_).')
    if len(db_name) > 64:
        raise exceptions.BadArgumentException('--hive-database-name', 'hive database name must be less than 64 characters.')
    return db_name