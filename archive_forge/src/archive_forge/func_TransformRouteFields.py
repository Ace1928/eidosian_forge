from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def TransformRouteFields(service_record):
    """Transforms a service's route fields into a marker class structure to print.

  Generates the custom printing format for a service's url, ingress, and traffic
  using the marker classes defined in custom_printer_base.

  Args:
    service_record: A Service object.

  Returns:
    A custom printer marker object describing the route fields print format.
  """
    no_status = service_record.status is None
    traffic_pairs = traffic_pair.GetTrafficTargetPairs(service_record.spec_traffic, service_record.status_traffic, service_record.is_managed, _INGRESS_UNSPECIFIED if no_status else service_record.status.latestReadyRevisionName)
    return _TransformTrafficPairs(traffic_pairs, '' if no_status else service_record.status.url, _GetIngress(service_record))