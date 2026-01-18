from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509, exceptions as cryptography_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive import verifiers
def verify_certificate_signature(signing_certificate, certificate):
    """Verify that the certificate was signed correctly.

    :param signing_certificate: the cryptography certificate object used to
           sign the certificate
    :param certificate: the cryptography certificate object that was signed
           by the signing certificate
    :raises: cryptography.exceptions.InvalidSignature if certificate signature
             verification fails.
    """
    signature_hash_algorithm = certificate.signature_hash_algorithm
    signature_bytes = certificate.signature
    signer_public_key = signing_certificate.public_key()
    if isinstance(signer_public_key, rsa.RSAPublicKey):
        verifier = verifiers.RSAVerifier(signature_bytes, signature_hash_algorithm, signer_public_key, padding.PKCS1v15())
    elif isinstance(signer_public_key, ec.EllipticCurvePublicKey):
        verifier = verifiers.ECCVerifier(signature_bytes, signature_hash_algorithm, signer_public_key)
    else:
        verifier = verifiers.DSAVerifier(signature_bytes, signature_hash_algorithm, signer_public_key)
    verifier.update(certificate.tbs_certificate_bytes)
    verifier.verify()