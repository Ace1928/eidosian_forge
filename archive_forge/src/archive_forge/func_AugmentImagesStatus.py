from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import policy
from googlecloudsdk.core import properties
def AugmentImagesStatus(self, resources, images):
    """Modify images status based on OrgPolicy."""
    return policy.AugmentImagesStatus(resources, properties.VALUES.core.project.GetOrFail(), images)