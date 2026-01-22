from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
def _create_certificates(self, root_dn=None, server_dn=None, client_dn=None):
    root_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='root')
    if root_dn:
        root_subj = unit.update_dn(root_subj, root_dn)
    root_cert, root_key = unit.create_certificate(root_subj)
    keystone_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='keystone.local')
    if server_dn:
        keystone_subj = unit.update_dn(keystone_subj, server_dn)
    ks_cert, ks_key = unit.create_certificate(keystone_subj, ca=root_cert, ca_key=root_key)
    client_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test')
    if client_dn:
        client_subj = unit.update_dn(client_subj, client_dn)
    client_cert, client_key = unit.create_certificate(client_subj, ca=root_cert, ca_key=root_key)
    return (root_cert, root_key, ks_cert, ks_key, client_cert, client_key)