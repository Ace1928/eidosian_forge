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
def _AddStorageLVPShareConfig(bare_metal_lvp_share_config_group):
    """Adds flags to set LVP Share class and path used by the storage.

  Args:
    bare_metal_lvp_share_config_group: The parent group to add the flags to.
  """
    bare_metal_storage_lvp_share_config_group = bare_metal_lvp_share_config_group.add_group(help=' LVP share class and path used by the storage.', required=True)
    bare_metal_storage_lvp_share_config_group.add_argument('--lvp-share-path', required=True, help='Path for the LVP share class.')
    bare_metal_storage_lvp_share_config_group.add_argument('--lvp-share-storage-class', required=True, help='Storage class for LVP share.')