from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.firewall_attachments import attachment_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddProducerForwardingRuleArg(parser, help_text='Path of a forwarding rule that points to a backend with GENEVE-capable VMs.'):
    parser.add_argument('--producer-forwarding-rule', required=True, help=help_text)