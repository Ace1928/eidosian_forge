from __future__ import annotations
import logging as _logging
import re as _re
from datetime import datetime as _datetime
from datetime import timezone
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union
from cryptography.exceptions import InvalidSignature as _InvalidSignature
from cryptography.hazmat.backends import default_backend as _default_backend
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey as _DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA as _ECDSA
from cryptography.hazmat.primitives.asymmetric.ec import (
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15 as _PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey as _RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.x448 import (
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from cryptography.hazmat.primitives.hashes import SHA1 as _SHA1
from cryptography.hazmat.primitives.hashes import Hash as _Hash
from cryptography.hazmat.primitives.serialization import Encoding as _Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat as _PublicFormat
from cryptography.x509 import AuthorityInformationAccess as _AuthorityInformationAccess
from cryptography.x509 import ExtendedKeyUsage as _ExtendedKeyUsage
from cryptography.x509 import ExtensionNotFound as _ExtensionNotFound
from cryptography.x509 import TLSFeature as _TLSFeature
from cryptography.x509 import TLSFeatureType as _TLSFeatureType
from cryptography.x509 import load_pem_x509_certificate as _load_pem_x509_certificate
from cryptography.x509.ocsp import OCSPCertStatus as _OCSPCertStatus
from cryptography.x509.ocsp import OCSPRequestBuilder as _OCSPRequestBuilder
from cryptography.x509.ocsp import OCSPResponseStatus as _OCSPResponseStatus
from cryptography.x509.ocsp import load_der_ocsp_response as _load_der_ocsp_response
from cryptography.x509.oid import (
from cryptography.x509.oid import ExtendedKeyUsageOID as _ExtendedKeyUsageOID
from requests import post as _post
from requests.exceptions import RequestException as _RequestException
from pymongo import _csot
def _get_certs_by_name(certificates: Iterable[Certificate], issuer: Certificate, responder_name: Optional[Name]) -> list[Certificate]:
    return [cert for cert in certificates if cert.subject == responder_name and cert.issuer == issuer.subject]