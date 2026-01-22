from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def ParseAvailabilityType(alloydb_messages, availability_type):
    if availability_type:
        return alloydb_messages.Instance.AvailabilityTypeValueValuesEnum.lookup_by_name(availability_type.upper())
    return None