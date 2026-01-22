from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.spanner import response_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ListGeneric(instance_partition, instance):
    """List operations on an instance partition with generic LRO API."""
    client = apis.GetClientInstance(_API_NAME, _API_VERSION)
    msgs = apis.GetMessagesModule(_API_NAME, _API_VERSION)
    ref = resources.REGISTRY.Parse(instance_partition, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'instancesId': instance}, collection='spanner.projects.instances.instancePartitions')
    req = msgs.SpannerProjectsInstancesInstancePartitionsOperationsListRequest(name=ref.RelativeName() + '/operations')
    return list_pager.YieldFromList(client.projects_instances_instancePartitions_operations, req, field='operations', batch_size_attribute='pageSize')