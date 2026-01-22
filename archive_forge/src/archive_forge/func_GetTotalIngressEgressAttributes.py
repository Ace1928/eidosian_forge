from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import dataclasses
from googlecloudsdk.api_lib.accesscontextmanager import levels as levels_api
from googlecloudsdk.api_lib.accesscontextmanager import zones as perimeters_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
def GetTotalIngressEgressAttributes(self, perimeters_to_display):
    """Returns total ingress/egress attributes quota usage.

    Args:
      perimeters_to_display: Response of ListServicePerimeters API
    """
    elements_count = 0
    for metric in perimeters_to_display:
        configs = []
        if metric.status:
            configs.append(metric.status)
        if metric.spec:
            configs.append(metric.spec)
        for config in configs:
            if config.ingressPolicies:
                for ingress_policy in config.ingressPolicies:
                    elements_count += len(ingress_policy.ingressFrom.sources)
                    elements_count += len(ingress_policy.ingressFrom.identities)
                    elements_count += sum((len(o.methodSelectors) for o in ingress_policy.ingressTo.operations))
                    elements_count += len(ingress_policy.ingressTo.resources)
            if config.egressPolicies:
                for egress_policy in config.egressPolicies:
                    elements_count += len(egress_policy.egressFrom.identities)
                    elements_count += sum((len(o.methodSelectors) for o in egress_policy.egressTo.operations))
                    elements_count += len(egress_policy.egressTo.resources)
    return elements_count