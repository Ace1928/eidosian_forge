from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _TransformNumInstances(resource):
    """Sums up number of instances in a patch job."""
    if 'instanceDetailsSummary' not in resource:
        return None
    instance_details_summary = resource['instanceDetailsSummary']
    num_instances = 0
    for status in instance_details_summary:
        num_instances += int(instance_details_summary.get(status, 0))
    return num_instances