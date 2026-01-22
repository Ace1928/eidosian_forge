from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_security import GetClientInstance
from googlecloudsdk.api_lib.network_security import GetMessagesModule
from googlecloudsdk.core import log
def ListOrganizationAddressGroupReferences(release_track, args):
    service = GetClientInstance(release_track).organizations_locations_addressGroups
    messages = GetMessagesModule(release_track)
    request_type = messages.NetworksecurityOrganizationsLocationsAddressGroupsListReferencesRequest
    return ListAddressGroupReferences(service, request_type, args)