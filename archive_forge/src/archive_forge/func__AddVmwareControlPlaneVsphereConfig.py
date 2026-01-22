from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddVmwareControlPlaneVsphereConfig(vmware_control_plane_node_config_group: parser_arguments.ArgumentInterceptor, release_track: base.ReleaseTrack=None):
    """Adds a flag for VmwareControlPlaneVsphereConfig message."""
    if release_track and release_track == base.ReleaseTrack.ALPHA:
        vmware_control_plane_node_config_group.add_argument('--control-plane-vsphere-config', help='Vsphere-specific configurations.', hidden=True, metavar='datastore=DATASTORE,storage-policy-name=STORAGE_POLICY_NAME', type=arg_parsers.ArgDict(spec={'datastore': str, 'storage-policy-name': str}))