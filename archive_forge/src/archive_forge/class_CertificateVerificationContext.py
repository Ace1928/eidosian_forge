from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509, exceptions as cryptography_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive import verifiers
class CertificateVerificationContext(object):
    """A collection of signing certificates.

    A collection of signing certificates that may be used to verify the
    signatures of other certificates.
    """

    def __init__(self, certificate_tuples, enforce_valid_dates=True, enforce_signing_extensions=True, enforce_path_length=True):
        self._signing_certificates = []
        for certificate_tuple in certificate_tuples:
            certificate_uuid, certificate = certificate_tuple
            if not isinstance(certificate, x509.Certificate):
                LOG.error('A signing certificate must be an x509.Certificate object.')
                continue
            if enforce_valid_dates:
                if not is_within_valid_dates(certificate):
                    LOG.warning("Certificate '%s' is outside its valid date range and cannot be used as a signing certificate.", certificate_uuid)
                    continue
            if enforce_signing_extensions:
                if not can_sign_certificates(certificate, certificate_uuid):
                    LOG.warning("Certificate '%s' is not configured to act as a signing certificate. It will not be used as a signing certificate.", certificate_uuid)
                    continue
            self._signing_certificates.append(certificate_tuple)
        self._signed_certificate = None
        self._enforce_valid_dates = enforce_valid_dates
        self._enforce_path_length = enforce_path_length

    def update(self, certificate):
        """Process the certificate to be verified.

        Raises an exception if the certificate is invalid. Stores it
        otherwise.

        :param certificate: the cryptography certificate to be verified
        :raises: SignatureVerificationError if the certificate is not of the
                 right type or if it is outside its valid date range.
        """
        if not isinstance(certificate, x509.Certificate):
            raise exception.SignatureVerificationError('The certificate must be an x509.Certificate object.')
        if self._enforce_valid_dates:
            if not is_within_valid_dates(certificate):
                raise exception.SignatureVerificationError('The certificate is outside its valid date range.')
        self._signed_certificate = certificate

    def verify(self):
        """Locate the certificate's signing certificate and verify it.

        Locate the certificate's signing certificate in the context
        certificate cache, using both subject/issuer name matching and
        signature verification. If the certificate is self-signed, verify that
        it is also located in the context's certificate cache. Construct the
        certificate chain from certificates in the context certificate cache.
        Verify that the signing certificate can have a sufficient number of
        child certificates to support the chain.

        :raises: SignatureVerificationError if certificate validation fails
                 for any reason, including mismatched signatures or a failure
                 to find the required signing certificate.
        """
        signed_certificate = self._signed_certificate
        certificate_chain = [('base', signed_certificate)]
        while True:
            signing_certificate_tuple = None
            for certificate_tuple in self._signing_certificates:
                _, candidate = certificate_tuple
                if is_issuer(candidate, signed_certificate):
                    signing_certificate_tuple = certificate_tuple
                    break
            if signing_certificate_tuple:
                if signed_certificate == signing_certificate_tuple[1]:
                    break
                else:
                    certificate_chain.insert(0, signing_certificate_tuple)
                    signed_certificate = signing_certificate_tuple[1]
            else:
                uuid = certificate_chain[0][0]
                raise exception.SignatureVerificationError('Certificate chain building failed. Could not locate the signing certificate for %s in the set of trusted certificates.' % 'the base certificate' if uuid == 'base' else "certificate '%s'" % uuid)
        if self._enforce_path_length:
            for i in range(len(certificate_chain)):
                certificate = certificate_chain[i][1]
                if certificate == certificate_chain[-1][1]:
                    break
                try:
                    constraints = certificate.extensions.get_extension_for_oid(x509.oid.ExtensionOID.BASIC_CONSTRAINTS).value
                except x509.extensions.ExtensionNotFound:
                    raise exception.SignatureVerificationError("Certificate validation failed. The signing certificate '%s' does not have a basic constraints extension." % certificate_chain[i][0])
                chain_length = len(certificate_chain[i:])
                chain_length = chain_length - 2 if chain_length > 2 else 0
                if constraints.path_length < chain_length:
                    raise exception.SignatureVerificationError("Certificate validation failed. The signing certificate '%s' is not configured to support certificate chains of sufficient length." % certificate_chain[i][0])