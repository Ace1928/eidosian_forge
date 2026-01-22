from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import secure_tags_utils
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetLocationPolicyLocationsMaxCountPerZoneFeatureDisabled(args, messages):
    """Helper function for getting location for location policy."""
    locations = []
    for zone, policy in args.location_policy.items():
        zone_policy = arg_utils.ChoiceToEnum(policy, messages.LocationPolicyLocation.PreferenceValueValuesEnum)
        locations.append(messages.LocationPolicy.LocationsValue.AdditionalProperty(key='zones/{}'.format(zone), value=messages.LocationPolicyLocation(preference=zone_policy)))
    return messages.LocationPolicy.LocationsValue(additionalProperties=locations)