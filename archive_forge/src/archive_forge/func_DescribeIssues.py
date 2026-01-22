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
def DescribeIssues(self, name, args=None):
    """Describe database entity issues in a conversion worksapce.

    Args:
      name: str, the name for conversion worksapce being described.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Issues found for the database entities of the conversion worksapce.
    """
    page_size = 4000
    entity_issues = self.DescribeIssuesHelper(name, page_size, args, self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.SOURCE_TREE)
    entity_issues.extend(self.DescribeIssuesHelper(name, page_size, args, self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.DRAFT_TREE))
    return entity_issues