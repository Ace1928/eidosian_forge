from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslSettings(_messages.Message):
    """SSL configuration for a DomainMapping resource.

  Enums:
    SslManagementTypeValueValuesEnum: SSL management type for this domain. If
      AUTOMATIC, a managed certificate is automatically provisioned. If
      MANUAL, certificate_id must be manually specified in order to configure
      SSL for this domain.

  Fields:
    certificateId: ID of the AuthorizedCertificate resource configuring SSL
      for the application. Clearing this field will remove SSL support.By
      default, a managed certificate is automatically created for every domain
      mapping. To omit SSL support or to configure SSL manually, specify
      SslManagementType.MANUAL on a CREATE or UPDATE request. You must be
      authorized to administer the AuthorizedCertificate resource to manually
      map it to a DomainMapping resource. Example: 12345.
    pendingManagedCertificateId: ID of the managed AuthorizedCertificate
      resource currently being provisioned, if applicable. Until the new
      managed certificate has been successfully provisioned, the previous SSL
      state will be preserved. Once the provisioning process completes, the
      certificate_id field will reflect the new managed certificate and this
      field will be left empty. To remove SSL support while there is still a
      pending managed certificate, clear the certificate_id field with an
      UpdateDomainMappingRequest.@OutputOnly
    sslManagementType: SSL management type for this domain. If AUTOMATIC, a
      managed certificate is automatically provisioned. If MANUAL,
      certificate_id must be manually specified in order to configure SSL for
      this domain.
  """

    class SslManagementTypeValueValuesEnum(_messages.Enum):
        """SSL management type for this domain. If AUTOMATIC, a managed
    certificate is automatically provisioned. If MANUAL, certificate_id must
    be manually specified in order to configure SSL for this domain.

    Values:
      AUTOMATIC: SSL support for this domain is configured automatically. The
        mapped SSL certificate will be automatically renewed.
      MANUAL: SSL support for this domain is configured manually by the user.
        Either the domain has no SSL support or a user-obtained SSL
        certificate has been explictly mapped to this domain.
    """
        AUTOMATIC = 0
        MANUAL = 1
    certificateId = _messages.StringField(1)
    pendingManagedCertificateId = _messages.StringField(2)
    sslManagementType = _messages.EnumField('SslManagementTypeValueValuesEnum', 3)