from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ListDatabaseOperations(instance, database=None, type_filter=None):
    """List database operations using the Cloud Spanner specific API."""
    client = apis.GetClientInstance('spanner', 'v1')
    msgs = apis.GetMessagesModule('spanner', 'v1')
    instance_ref = resources.REGISTRY.Parse(instance, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='spanner.projects.instances')
    if database:
        return List(instance, database, type_filter)
    req = msgs.SpannerProjectsInstancesDatabaseOperationsListRequest(parent=instance_ref.RelativeName(), filter=type_filter)
    return list_pager.YieldFromList(client.projects_instances_databaseOperations, req, field='operations', batch_size_attribute='pageSize')