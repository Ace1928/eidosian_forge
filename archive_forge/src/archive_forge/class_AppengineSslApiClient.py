from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
class AppengineSslApiClient(base.AppengineApiClientBase):
    """Client used by gcloud to communicate with the App Engine SSL APIs."""

    def __init__(self, client):
        base.AppengineApiClientBase.__init__(self, client)
        self._registry = resources.REGISTRY.Clone()
        self._registry.RegisterApiByName('appengine', client._VERSION)

    def CreateSslCertificate(self, display_name, cert_path, private_key_path):
        """Creates a certificate for the given application.

    Args:
      display_name: str, the display name for the new certificate.
      cert_path: str, location on disk to a certificate file.
      private_key_path: str, location on disk to a private key file.

    Returns:
      The created AuthorizedCertificate object.

    Raises:
      Error if the file does not exist or can't be opened/read.
    """
        certificate_data = files.ReadFileContents(cert_path)
        private_key_data = files.ReadFileContents(private_key_path)
        cert = self.messages.CertificateRawData(privateKey=private_key_data, publicCertificate=certificate_data)
        auth_cert = self.messages.AuthorizedCertificate(displayName=display_name, certificateRawData=cert)
        request = self.messages.AppengineAppsAuthorizedCertificatesCreateRequest(parent=self._FormatApp(), authorizedCertificate=auth_cert)
        return self.client.apps_authorizedCertificates.Create(request)

    def DeleteSslCertificate(self, cert_id):
        """Deletes an authorized certificate for the given application.

    Args:
      cert_id: str, the id of the certificate to delete.
    """
        request = self.messages.AppengineAppsAuthorizedCertificatesDeleteRequest(name=self._FormatSslCert(cert_id))
        self.client.apps_authorizedCertificates.Delete(request)

    def GetSslCertificate(self, cert_id):
        """Gets a certificate for the given application.

    Args:
      cert_id: str, the id of the certificate to retrieve.

    Returns:
      The retrieved AuthorizedCertificate object.
    """
        request = self.messages.AppengineAppsAuthorizedCertificatesGetRequest(name=self._FormatSslCert(cert_id), view=self.messages.AppengineAppsAuthorizedCertificatesGetRequest.ViewValueValuesEnum.FULL_CERTIFICATE)
        return self.client.apps_authorizedCertificates.Get(request)

    def ListSslCertificates(self):
        """Lists all authorized certificates for the given application.

    Returns:
      A list of AuthorizedCertificate objects.
    """
        request = self.messages.AppengineAppsAuthorizedCertificatesListRequest(parent=self._FormatApp())
        response = self.client.apps_authorizedCertificates.List(request)
        return response.certificates

    def UpdateSslCertificate(self, cert_id, display_name=None, cert_path=None, private_key_path=None):
        """Updates a certificate for the given application.

    One of display_name, cert_path, or private_key_path should be set. Omitted
    fields will not be updated from their current value. Any invalid arguments
    will fail the entire command.

    Args:
      cert_id: str, the id of the certificate to update.
      display_name: str, the display name for a new certificate.
      cert_path: str, location on disk to a certificate file.
      private_key_path: str, location on disk to a private key file.

    Returns:
      The created AuthorizedCertificate object.

    Raises: InvalidInputError if the user does not specify both cert and key.
    """
        if bool(cert_path) ^ bool(private_key_path):
            missing_arg = '--certificate' if not cert_path else '--private-key'
            raise exceptions.RequiredArgumentException(missing_arg, 'The certificate and the private key must both be updated together.')
        mask_fields = []
        if display_name:
            mask_fields.append('displayName')
        cert_data = None
        if cert_path and private_key_path:
            certificate = files.ReadFileContents(cert_path)
            private_key = files.ReadFileContents(private_key_path)
            cert_data = self.messages.CertificateRawData(privateKey=private_key, publicCertificate=certificate)
            mask_fields.append('certificateRawData')
        auth_cert = self.messages.AuthorizedCertificate(displayName=display_name, certificateRawData=cert_data)
        if not mask_fields:
            raise exceptions.MinimumArgumentException(['--certificate', '--private-key', '--display-name'], 'Please specify at least one attribute to the certificate update.')
        request = self.messages.AppengineAppsAuthorizedCertificatesPatchRequest(name=self._FormatSslCert(cert_id), authorizedCertificate=auth_cert, updateMask=','.join(mask_fields))
        return self.client.apps_authorizedCertificates.Patch(request)

    def _FormatSslCert(self, cert_id):
        res = self._registry.Parse(cert_id, params={'appsId': self.project}, collection='appengine.apps.authorizedCertificates')
        return res.RelativeName()