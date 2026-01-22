from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def GetKrmApiHost(name):
    """Fetches a KRMApiHost instance, and returns it as a messages.KrmApiHost.

  Calls into the GetKrmApiHosts API.

  Args:
    name: the fully qualified name of the instance, e.g.
      "projects/p/locations/l/krmApiHosts/k".

  Returns:
    A messages.KrmApiHost.

  Raises:
    HttpNotFoundError: if the instance didn't exist.
  """
    client = GetClientInstance()
    messages = client.MESSAGES_MODULE
    return client.projects_locations_krmApiHosts.Get(messages.KrmapihostingProjectsLocationsKrmApiHostsGetRequest(name=name))