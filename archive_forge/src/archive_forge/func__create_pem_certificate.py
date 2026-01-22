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
def _create_pem_certificate(self, subject_dn, ca=None, ca_key=None):
    cert, _ = self._create_certificate(subject_dn, ca=ca, ca_key=ca_key)
    return cert.public_bytes(Encoding.PEM)