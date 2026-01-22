from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1 import appengine_v1_messages as messages
Lists all domains the user is authorized to administer.

      Args:
        request: (AppengineProjectsLocationsApplicationsAuthorizedDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedDomainsResponse) The response message.
      