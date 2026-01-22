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
def AddWindowsOsVersionFlag(parser, hidden=False):
    """Adds --windows-os-version flag to the given parser.

  Flag:
  --windows-os-version={ltsc2019|ltsc2022}

  Args:
    parser: A given parser.
    hidden: Indicates that the flags are hidden.
  """
    help_text = '\n    Specifies the Windows Server Image to use when creating a Windows node pool.\n    Valid variants can be "ltsc2019", "ltsc2022". It means using LTSC2019 server\n    image or LTSC2022 server image. If the node pool doesn\'t specify a Windows\n    Server Image Os version, then Ltsc2019 will be the default one to use.\n  '
    parser.add_argument('--windows-os-version', help=help_text, hidden=hidden, choices=['ltsc2019', 'ltsc2022'], default=None)