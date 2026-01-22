from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def InterconnectAttachmentArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='interconnect attachment', completer=InterconnectAttachmentsCompleter, plural=plural, required=required, regional_collection='compute.interconnectAttachments', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)