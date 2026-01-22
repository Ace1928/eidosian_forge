from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetInterconnectAttachmentRef(resources, name, region, project):
    """Generates an interconnect attachment reference from the specified name, region and project."""
    return resources.Parse(name, collection='compute.interconnectAttachments', params={'project': project, 'region': region})