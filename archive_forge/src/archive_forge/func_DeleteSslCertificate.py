from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def DeleteSslCertificate(self, cert_id):
    """Deletes an authorized certificate for the given application.

    Args:
      cert_id: str, the id of the certificate to delete.
    """
    request = self.messages.AppengineAppsAuthorizedCertificatesDeleteRequest(name=self._FormatSslCert(cert_id))
    self.client.apps_authorizedCertificates.Delete(request)