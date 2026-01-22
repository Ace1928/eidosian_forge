from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _GetIngress(record):
    """Gets the ingress traffic allowed to call the service."""
    if platforms.GetPlatform() == platforms.PLATFORM_MANAGED:
        spec_ingress = record.annotations.get(service.INGRESS_ANNOTATION)
        status_ingress = record.annotations.get(service.INGRESS_STATUS_ANNOTATION)
        if spec_ingress == status_ingress:
            return spec_ingress
        else:
            spec_ingress = spec_ingress or _INGRESS_UNSPECIFIED
            status_ingress = status_ingress or _INGRESS_UNSPECIFIED
            return '{} (currently {})'.format(spec_ingress, status_ingress)
    elif record.labels.get(service.ENDPOINT_VISIBILITY) == service.CLUSTER_LOCAL:
        return service.INGRESS_INTERNAL
    else:
        return service.INGRESS_ALL