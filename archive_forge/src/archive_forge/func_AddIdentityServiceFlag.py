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
def AddIdentityServiceFlag(parser):
    parser.add_argument('--enable-identity-service', default=None, action='store_true', help='Enable Identity Service component on the cluster.\n\nWhen enabled, users can authenticate to Kubernetes cluster with external\nidentity providers.\n\nIdentity Service is by default disabled when creating a new cluster.\nTo disable Identity Service in an existing cluster, explicitly set flag\n`--no-enable-identity-service`.\n')