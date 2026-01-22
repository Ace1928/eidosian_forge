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
def _GetSourceInstanceTemplate(args, resources, instance_template_resource):
    """Get sourceInstanceTemplate value as required by API."""
    if not args.IsSpecified('source_instance_template'):
        return None
    ref = instance_template_resource.ResolveAsResource(args, resources)
    return ref.SelfLink()