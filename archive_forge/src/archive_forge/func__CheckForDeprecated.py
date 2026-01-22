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
def _CheckForDeprecated(self, image):
    deprecated = False
    deprecate_info = image.get('deprecated')
    if deprecate_info is not None:
        image_state = deprecate_info.get('state')
        if image_state and image_state != 'ACTIVE':
            deprecated = True
    return deprecated