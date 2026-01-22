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
def DescribeDDLs(self, name, args=None):
    """Describe DDLs in a conversion worksapce.

    Args:
      name: str, the name for conversion worksapce being described.
      args: argparse.Namespace, the arguments that this command was invoked
        with.

    Returns:
      DDLs for the entities of the conversion worksapce.
    """
    entity_ddls = []
    page_size = 4000
    describe_req = self._GetDescribeDDLsRequest(name, page_size, str(), args)
    while True:
        response = self._service.DescribeDatabaseEntities(describe_req)
        for entity in response.databaseEntities:
            for entity_ddl in entity.entityDdl:
                entity_ddls.append({'ddl': entity_ddl.ddl})
        if not response.nextPageToken:
            break
        describe_req.pageToken = response.nextPageToken
    return entity_ddls