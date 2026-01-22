from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.command_lib.container.fleet import agent_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import exclusivity_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _InstallOrUpdateExclusivityArtifacts(self, kube_client, membership_ref):
    """Install the exclusivity artifacts for the cluster.

    Update the exclusivity artifacts if a new version is available if the
    cluster has already being registered.

    Args:
      kube_client: a KubernetesClient
      membership_ref: the full resource name of the membership the cluster is
        registered with.

    Raises:
      apitools.base.py.HttpError: if the API request returns an HTTP error.
      exceptions.Error: if the kubectl interation with the cluster failed.
    """
    crd_manifest = kube_client.GetMembershipCRD()
    cr_manifest = kube_client.GetMembershipCR() if crd_manifest else ''
    res = api_util.GenerateExclusivityManifest(crd_manifest, cr_manifest, membership_ref)
    kube_client.ApplyMembership(res.crdManifest, res.crManifest)