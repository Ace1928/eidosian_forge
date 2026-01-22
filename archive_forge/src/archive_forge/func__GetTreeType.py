from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def _GetTreeType(self, tree_type):
    """Returns the tree type for database entities."""
    if tree_type == 'SOURCE':
        return self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.SOURCE_TREE
    if tree_type == 'DRAFT':
        return self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.DRAFT_TREE
    return self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.DB_TREE_TYPE_UNSPECIFIED