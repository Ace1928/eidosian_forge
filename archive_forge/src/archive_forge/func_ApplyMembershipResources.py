from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.core import exceptions
def ApplyMembershipResources(kube_client, project):
    """Creates or updates the Membership CRD and CR with the hub project id.

  Args:
    kube_client: A KubernetesClient.
    project: The project id of the hub the cluster is a member of.

  Raises:
    exceptions.Error: if the Membership CR or CRD couldn't be applied.
  """
    membership_cr_manifest = MEMBERSHIP_CR_TEMPLATE.format(project_id=project)
    kube_client.ApplyMembership(MEMBERSHIP_CRD_MANIFEST, membership_cr_manifest)