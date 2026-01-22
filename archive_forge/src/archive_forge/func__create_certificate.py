import base64
import datetime
import hashlib
import os
import ssl
import uuid
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography import x509
import fixtures
from keystoneauth1 import fixture
from oslo_serialization import jsonutils
import testresources
def _create_certificate(self, subject_dn, ca=None, ca_key=None):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = ca.subject if ca else subject_dn
    if not ca_key:
        ca_key = private_key
    today = datetime.datetime.today()
    cert = x509.CertificateBuilder(issuer_name=issuer, subject_name=subject_dn, public_key=private_key.public_key(), serial_number=x509.random_serial_number(), not_valid_before=today, not_valid_after=today + datetime.timedelta(365, 0, 0)).sign(ca_key, hashes.SHA256())
    return (cert, private_key)