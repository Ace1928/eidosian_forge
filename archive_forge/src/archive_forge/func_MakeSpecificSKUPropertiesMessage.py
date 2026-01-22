from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeSpecificSKUPropertiesMessage(messages, instance_properties, total_count, source_instance_template_ref=None):
    """Constructs a specific sku properties message object."""
    properties = None
    source_instance_template_url = None
    if source_instance_template_ref:
        source_instance_template_url = source_instance_template_ref.SelfLink()
    else:
        properties = instance_properties
    return messages.FutureReservationSpecificSKUProperties(totalCount=total_count, sourceInstanceTemplate=source_instance_template_url, instanceProperties=properties)