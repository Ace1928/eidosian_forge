from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def InterconnectAttachmentArgumentForRouter(required=False, plural=False, operation_type='added'):
    resource_name = 'interconnectAttachment{0}'.format('s' if plural else '')
    return compute_flags.ResourceArgument(resource_name=resource_name, name='--interconnect-attachment', completer=InterconnectAttachmentsCompleter, plural=plural, required=required, regional_collection='compute.interconnectAttachments', short_help='The interconnect attachment of the interface being {0}.'.format(operation_type), region_explanation='If not specified it will be set to the region of the router.')