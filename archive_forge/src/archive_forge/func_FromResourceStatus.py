from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
@classmethod
def FromResourceStatus(cls, cluster_name, resource):
    """Initialize a ListItem object from a resourceStatus.

    Args:
      cluster_name: name of the cluster the results are from
      resource: individual resource status dictionary parsed from kubectl

    Returns:
      new instance of ListItem
    """
    condition = ''
    reconcile_condition = utils.GetActuationCondition(resource)
    conditions = resource.get('conditions', [])[:]
    if reconcile_condition:
        conditions.insert(0, reconcile_condition)
    if conditions:
        delimited_msg = ', '.join(["'{}'".format(c['message']) for c in conditions])
        condition = '[{}]'.format(delimited_msg)
    return cls(cluster_name=cluster_name, group=resource['group'], kind=resource['kind'], namespace=resource['namespace'], name=resource['name'], status=resource['status'], condition=condition)