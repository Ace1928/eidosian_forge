from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddMeshCertificatesFlags(parser):
    """Adds Mesh Certificates flags to the parser."""
    parser.add_argument('--enable-mesh-certificates', default=None, hidden=True, action='store_true', help=textwrap.dedent('    Enable Mesh Certificates.\n\n    After the cluster is created, configure an issuing certificate authority using\n    the Kubernetes API.\n\n    To disable Mesh Certificates in an existing cluster, explicitly set flag\n    `--no-enable-mesh-certificates`.\n    '))