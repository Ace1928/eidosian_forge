from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetBindedWorkloads(self, resource: runapps_v1alpha1_messages.Resource, all_resources: List[runapps_v1alpha1_messages.Resource], workload_type: str='service') -> List[runapps_v1alpha1_messages.ResourceID]:
    """Returns list of workloads that are associated to this resource.

    If the resource is a backing service, then it returns a list of workloads
    binding to the resource. If the resource is an ingress service, then all
    of the workloads it is binding to.

    Args:
      resource: the resource object of the integration.
      all_resources: all the resources in the application.
      workload_type: type of the workload to search for.

    Returns:
      list ResourceID of the binded workloads.
    """
    if self.is_backing_service:
        filtered_workloads = [res for res in all_resources if res.id.type == workload_type]
        return [workload.id.name for workload in filtered_workloads if FindBindings(workload, resource.id.type, resource.id.name)]
    return [res_id.targetRef.id.name for res_id in FindBindingsRecursive(resource, workload_type)]