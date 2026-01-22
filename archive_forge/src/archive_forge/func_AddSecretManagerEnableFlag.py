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
def AddSecretManagerEnableFlag(parser, hidden=True):
    """Adds --enable-secret-manager flag to the given parser.

  Args:
    parser: A given parser.
    hidden: hidden status
  """
    help_text = '        Enables the Secret Manager CSI driver provider component. See\n        https://secrets-store-csi-driver.sigs.k8s.io/introduction\n        https://github.com/GoogleCloudPlatform/secrets-store-csi-driver-provider-gcp\n\n        To disable in an existing cluster, explicitly set flag to\n        --no-enable-secret-manager\n    '
    parser.add_argument('--enable-secret-manager', action='store_true', default=None, help=help_text, hidden=hidden)