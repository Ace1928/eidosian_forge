from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_security import GetClientInstance
from googlecloudsdk.api_lib.network_security import GetMessagesModule
from googlecloudsdk.core import log
def ListProjectAddressGroupReferences(release_track, args):
    service = GetClientInstance(release_track).projects_locations_addressGroups
    messages = GetMessagesModule(release_track)
    request_type = messages.NetworksecurityProjectsLocationsAddressGroupsListReferencesRequest
    return ListAddressGroupReferences(service, request_type, args)