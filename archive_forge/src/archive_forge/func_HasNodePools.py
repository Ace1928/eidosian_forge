from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def HasNodePools(self, cluster_ref):
    """Checks if the cluster has a node pool."""
    req = self._service.GetRequestType('List')(parent=cluster_ref.RelativeName(), pageSize=1)
    res = self._service.List(req)
    node_pools = getattr(res, self._list_result_field, None)
    return True if node_pools else False