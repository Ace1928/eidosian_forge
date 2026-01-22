from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeResourcePolicies(messages, reservation_ref, resource_policy_dictionary, resources):
    """Constructs the resource policies message objects."""
    if resource_policy_dictionary is None:
        return None
    return messages.Reservation.ResourcePoliciesValue(additionalProperties=[messages.Reservation.ResourcePoliciesValue.AdditionalProperty(key=key, value=MakeUrl(resources, value, reservation_ref)) for key, value in sorted(six.iteritems(resource_policy_dictionary))])