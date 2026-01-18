from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.core import resources
Deletes a private connection.

    Args:
      private_connection_name: str, the name of the resource to delete.

    Returns:
      Operation: the operation for deleting the private connection.
    