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
def DescribeIssuesHelper(self, name, page_size, args, tree_type):
    """Describe issues in a conversion worksapce."""
    entity_issues = []
    describe_req = self._GetDescribeIssuesRequest(name, page_size, str(), args, tree_type)
    while True:
        response = self._service.DescribeDatabaseEntities(describe_req)
        for entity in response.databaseEntities:
            entity_issues.extend(self.GetIssuesHelper(entity))
        if not response.nextPageToken:
            break
        describe_req.pageToken = response.nextPageToken
    return entity_issues