from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
Updates a migration job.

    Args:
      name: str, the reference of the migration job to
          update.
      source_ref: a Resource reference to a
        datamigration.projects.locations.connectionProfiles resource.
      destination_ref: a Resource reference to a
        datamigration.projects.locations.connectionProfiles resource.
      args: argparse.Namespace, The arguments that this command was
          invoked with.

    Returns:
      Operation: the operation for updating the migration job.678888888
    