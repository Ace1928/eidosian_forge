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
def AddBasicAuthFlags(parser):
    """Adds basic auth flags to the given parser.

  Basic auth flags are: --username, --enable-basic-auth, and --password.

  Args:
    parser: A given parser.
  """
    basic_auth_group = parser.add_group(help='Basic auth')
    username_group = basic_auth_group.add_group(mutex=True, help='Options to specify the username.')
    username_help_text = 'The user name to use for basic auth for the cluster. Use `--password` to specify\na password; if not, the server will randomly generate one.'
    username_group.add_argument('--username', '-u', help=username_help_text)
    enable_basic_auth_help_text = 'Enable basic (username/password) auth for the cluster.  `--enable-basic-auth` is\nan alias for `--username=admin`; `--no-enable-basic-auth` is an alias for\n`--username=""`. Use `--password` to specify a password; if not, the server will\nrandomly generate one. For cluster versions before 1.12, if neither\n`--enable-basic-auth` nor `--username` is specified, `--enable-basic-auth` will\ndefault to `true`. After 1.12, `--enable-basic-auth` will default to `false`.'
    username_group.add_argument('--enable-basic-auth', help=enable_basic_auth_help_text, action='store_true', default=None)
    basic_auth_group.add_argument('--password', help='The password to use for cluster auth. Defaults to a server-specified randomly-generated string.')