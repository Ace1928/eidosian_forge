from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def ListVerifiedDomains(self):
    """Lists all domains verified by the current user.

    Returns:
      A list of AuthorizedDomain objects.
    """
    request = self.messages.AppengineAppsAuthorizedDomainsListRequest(parent=self._FormatApp())
    response = self.client.apps_authorizedDomains.List(request)
    return response.domains