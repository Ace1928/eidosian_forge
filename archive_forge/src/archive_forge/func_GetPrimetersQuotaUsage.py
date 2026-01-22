from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import dataclasses
from googlecloudsdk.api_lib.accesscontextmanager import levels as levels_api
from googlecloudsdk.api_lib.accesscontextmanager import zones as perimeters_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
def GetPrimetersQuotaUsage(self, perimeters_to_display):
    """Returns service primeters quota usage.

    Args:
      perimeters_to_display: Response of ListServicePerimeters API
    """
    arguments = list(perimeters_to_display)
    service_primeters = len(arguments)
    protected_resources = 0
    ingress_rules = 0
    egress_rules = 0
    total_ingress_egress_attributes = self.GetTotalIngressEgressAttributes(arguments)
    for metric in arguments:
        configs = []
        if metric.status:
            configs.append(metric.status)
        if metric.spec:
            configs.append(metric.spec)
        for config in configs:
            protected_resources += len(config.resources)
            ingress_rules += len(config.ingressPolicies)
            egress_rules += len(config.egressPolicies)
    return [Metric('Service primeters', service_primeters), Metric('Protected resources', protected_resources), Metric('Ingress rules', ingress_rules), Metric('Egress rules', egress_rules), Metric('Total ingress/egress attributes', total_ingress_egress_attributes)]