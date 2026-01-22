from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def CreateKrmApiHost(parent, krm_api_host_id, krm_api_host):
    """Creates a KRMApiHost instance, and returns the creation Operation.

  Calls into the CreateKrmApiHost API.

  Args:
    parent: the fully qualified name of the parent, e.g.
      "projects/p/locations/l".
    krm_api_host_id: the ID of the krmApiHost, e.g. "my-cluster" in
      "projects/p/locations/l/krmApiHosts/my-cluster".
    krm_api_host: a messages.KrmApiHost resource (containing properties like
      the bundle config)

  Returns:
    A messages.OperationMetadata representing the long-running operation.
  """
    client = GetClientInstance()
    messages = client.MESSAGES_MODULE
    return client.projects_locations_krmApiHosts.Create(messages.KrmapihostingProjectsLocationsKrmApiHostsCreateRequest(parent=parent, krmApiHost=krm_api_host, krmApiHostId=krm_api_host_id))