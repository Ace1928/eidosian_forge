from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def TransformInstanceSplitFields(worker_record):
    """Transforms a worker's instance split fields into a marker class structure to print.

  Generates the custom printing format for a worker's instance split using the
  marker classes defined in custom_printer_base.

  Args:
    worker_record: A Worker object.

  Returns:
    A custom printer marker object describing the instance split fields
    print format.
  """
    no_status = worker_record.status is None
    instance_split_pairs = traffic_pair.GetTrafficTargetPairs(worker_record.spec_traffic, worker_record.status_traffic, True, _LATEST_READY_REV_UNSPECIFIED if no_status else worker_record.status.latestReadyRevisionName)
    return _TransformInstanceSplitPairs(instance_split_pairs)