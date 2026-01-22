from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateAssuredWorkloadsParent(organization_id, location, workload_id):
    return 'organizations/{}/locations/{}/workloads/{}'.format(organization_id, location, workload_id)