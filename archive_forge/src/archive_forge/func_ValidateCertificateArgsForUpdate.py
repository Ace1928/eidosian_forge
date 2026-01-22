from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ValidateCertificateArgsForUpdate(certificate_id, no_certificate, certificate_management):
    ValidateCertificateArgs(certificate_id, certificate_management)
    if certificate_management:
        if certificate_management.upper() == 'AUTOMATIC' and no_certificate:
            raise exceptions.InvalidArgumentException('no-certificate-id', NO_CERTIFICATE_ID_MESSAGE)
        elif certificate_management.upper() == 'MANUAL' and (not certificate_id) and (not no_certificate):
            raise exceptions.InvalidArgumentException('certificate-id', 'A certificate ID or no-certificate must be provided when using manual certificate management.')