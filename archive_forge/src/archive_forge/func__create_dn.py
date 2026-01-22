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
def _create_dn(self, common_name=None, locality_name=None, state_or_province_name=None, organization_name=None, organizational_unit_name=None, country_name=None, street_address=None, domain_component=None, user_id=None, email_address=None):
    oid = x509.NameOID
    attr = x509.NameAttribute
    dn = []
    if common_name:
        dn.append(attr(oid.COMMON_NAME, common_name))
    if locality_name:
        dn.append(attr(oid.LOCALITY_NAME, locality_name))
    if state_or_province_name:
        dn.append(attr(oid.STATE_OR_PROVINCE_NAME, state_or_province_name))
    if organization_name:
        dn.append(attr(oid.ORGANIZATION_NAME, organization_name))
    if organizational_unit_name:
        dn.append(attr(oid.ORGANIZATIONAL_UNIT_NAME, organizational_unit_name))
    if country_name:
        dn.append(attr(oid.COUNTRY_NAME, country_name))
    if street_address:
        dn.append(attr(oid.STREET_ADDRESS, street_address))
    if domain_component:
        dn.append(attr(oid.DOMAIN_COMPONENT, domain_component))
    if user_id:
        dn.append(attr(oid.USER_ID, user_id))
    if email_address:
        dn.append(attr(oid.EMAIL_ADDRESS, email_address))
    return x509.Name(dn)