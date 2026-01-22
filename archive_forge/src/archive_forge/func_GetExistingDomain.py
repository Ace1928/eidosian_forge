from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
def GetExistingDomain(domain_ref):
    """Fetch existing AD domain."""
    client = util.GetClientForResource(domain_ref)
    messages = util.GetMessagesForResource(domain_ref)
    get_req = messages.ManagedidentitiesProjectsLocationsGlobalDomainsGetRequest(name=domain_ref.RelativeName())
    return client.projects_locations_global_domains.Get(get_req)