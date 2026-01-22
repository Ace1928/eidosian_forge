from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddEnableMultiNICConfig(bare_metal_network_config_group, is_update=False):
    if is_update:
        return None
    multi_nic_config_group = bare_metal_network_config_group.add_group(help='Multiple networking interfaces cluster configurations.')
    multi_nic_config_group.add_argument('--enable-multi-nic-config', action='store_true', help='If set, enable multiple network interfaces for your pods.')