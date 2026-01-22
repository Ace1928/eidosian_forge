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
def _GetPerInstanceProperties(args, messages, instance_names, support_custom_hostnames):
    """Helper function for getting per_instance_properties."""
    per_instance_hostnames = {}
    if support_custom_hostnames and args.IsSpecified('per_instance_hostnames'):
        per_instance_hostnames = args.per_instance_hostnames
    per_instance_properties = {}
    for name in instance_names:
        if name in per_instance_hostnames:
            per_instance_properties[name] = messages.BulkInsertInstanceResourcePerInstanceProperties(hostname=per_instance_hostnames[name])
        else:
            per_instance_properties[name] = messages.BulkInsertInstanceResourcePerInstanceProperties()
    return encoding.DictToAdditionalPropertyMessage(per_instance_properties, messages.BulkInsertInstanceResource.PerInstancePropertiesValue)