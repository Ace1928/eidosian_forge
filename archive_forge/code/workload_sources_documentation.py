from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List, Optional
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import identity_pool_waiter
from googlecloudsdk.core import resources as sdkresources
Make API calls to poll for a workload source LRO.

  Args:
    client: the iam v1 client.
    lro_ref: the lro ref returned from a LRO workload source API call.
    for_managed_identity: whether the workload source LRO is under a managed
      identity
    delete: whether it's a delete operation

  Returns:
    The result workload source or None for delete
  