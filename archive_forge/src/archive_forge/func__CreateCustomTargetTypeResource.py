from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CreateCustomTargetTypeResource(messages, name, project, region):
    """Creates custom target type resource with full name and the resource reference."""
    resource = messages.CustomTargetType()
    resource_ref = resources.REGISTRY.Parse(name, collection='clouddeploy.projects.locations.customTargetTypes', params={'projectsId': project, 'locationsId': region, 'customTargetTypesId': name})
    resource.name = resource_ref.RelativeName()
    return (resource, resource_ref)