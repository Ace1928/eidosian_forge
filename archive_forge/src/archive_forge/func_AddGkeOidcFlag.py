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
def AddGkeOidcFlag(parser):
    parser.add_argument('--enable-gke-oidc', default=None, action=actions.DeprecationAction('--enable-gke-oidc', warn='GKE OIDC is being replaced by Identity Service across Anthos and GKE. Thus, flag `--enable-gke-oidc` is also deprecated. Please use `--enable-identity-service` to enable the Identity Service component', action='store_true'), help='Enable GKE OIDC authentication on the cluster.\n\nWhen enabled, users would be able to authenticate to Kubernetes cluster after\nproperly setting OIDC config.\n\nGKE OIDC is by default disabled when creating a new cluster. To disable GKE OIDC\nin an existing cluster, explicitly set flag `--no-enable-gke-oidc`.\n')