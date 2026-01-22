from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.core import exceptions
def GetMembershipCROwnerID(kube_client):
    """Returns the project id of the fleet the cluster is a member of.

  The Membership Custom Resource stores the project id of the fleet the cluster
  is registered to in the `.spec.owner.id` field.

  Args:
    kube_client: A KubernetesClient.

  Returns:
    a string, the project id
    None, if the Membership CRD or CR do not exist on the cluster.

  Raises:
    exceptions.Error: if the Membership resource does not have a valid owner id
  """
    owner_id = kube_client.GetMembershipOwnerID()
    if owner_id is None:
        return None
    id_prefix = 'projects/'
    if not owner_id.startswith(id_prefix):
        raise exceptions.Error('Membership .spec.owner.id is invalid: {}'.format(owner_id))
    return owner_id[len(id_prefix):]