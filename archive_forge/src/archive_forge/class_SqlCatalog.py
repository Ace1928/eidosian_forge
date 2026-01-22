import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
class SqlCatalog(SqlTests, catalog_tests.CatalogTests):
    _legacy_endpoint_id_in_endpoint = True
    _enabled_default_to_true_when_creating_endpoint = True

    def test_get_v3_catalog_project_non_exist(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        malformed_url = 'http://192.168.1.104:8774/v2/$(project)s'
        endpoint = unit.new_endpoint_ref(service_id=service['id'], url=malformed_url, region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint.copy())
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.catalog_api.get_v3_catalog, 'fake-user', 'fake-project')

    def test_get_v3_catalog_with_empty_public_url(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(url='', service_id=service['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint.copy())
        catalog = PROVIDERS.catalog_api.get_v3_catalog(self.user_foo['id'], self.project_bar['id'])
        catalog_endpoint = catalog[0]
        self.assertEqual(service['name'], catalog_endpoint['name'])
        self.assertEqual(service['id'], catalog_endpoint['id'])
        self.assertEqual([], catalog_endpoint['endpoints'])

    def test_create_endpoint_region_returns_not_found(self):
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        endpoint = unit.new_endpoint_ref(region_id=uuid.uuid4().hex, service_id=service['id'])
        self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.create_endpoint, endpoint['id'], endpoint.copy())

    def test_create_region_invalid_id(self):
        region = unit.new_region_ref(id='0' * 256)
        self.assertRaises(exception.StringLengthExceeded, PROVIDERS.catalog_api.create_region, region)

    def test_create_region_invalid_parent_id(self):
        region = unit.new_region_ref(parent_region_id='0' * 256)
        self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.create_region, region)

    def test_delete_region_with_endpoint(self):
        region = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(region)
        child_region = unit.new_region_ref(parent_region_id=region['id'])
        PROVIDERS.catalog_api.create_region(child_region)
        service = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service['id'], service)
        child_endpoint = unit.new_endpoint_ref(region_id=child_region['id'], service_id=service['id'])
        PROVIDERS.catalog_api.create_endpoint(child_endpoint['id'], child_endpoint)
        self.assertRaises(exception.RegionDeletionError, PROVIDERS.catalog_api.delete_region, child_region['id'])
        endpoint = unit.new_endpoint_ref(region_id=region['id'], service_id=service['id'])
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        self.assertRaises(exception.RegionDeletionError, PROVIDERS.catalog_api.delete_region, region['id'])

    def test_v3_catalog_domain_scoped_token(self):
        srv_1 = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(srv_1['id'], srv_1)
        endpoint_1 = unit.new_endpoint_ref(service_id=srv_1['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint_1['id'], endpoint_1)
        srv_2 = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(srv_2['id'], srv_2)
        endpoint_2 = unit.new_endpoint_ref(service_id=srv_2['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint_2['id'], endpoint_2)
        self.config_fixture.config(group='endpoint_filter', return_all_endpoints_if_no_filter=True)
        catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(uuid.uuid4().hex, None)
        self.assertThat(catalog_ref, matchers.HasLength(2))
        self.config_fixture.config(group='endpoint_filter', return_all_endpoints_if_no_filter=False)
        catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(uuid.uuid4().hex, None)
        self.assertThat(catalog_ref, matchers.HasLength(0))

    def test_v3_catalog_endpoint_filter_enabled(self):
        srv_1 = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(srv_1['id'], srv_1)
        endpoint_1 = unit.new_endpoint_ref(service_id=srv_1['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint_1['id'], endpoint_1)
        endpoint_2 = unit.new_endpoint_ref(service_id=srv_1['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint_2['id'], endpoint_2)
        PROVIDERS.catalog_api.add_endpoint_to_project(endpoint_1['id'], self.project_bar['id'])
        catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(uuid.uuid4().hex, self.project_bar['id'])
        self.assertThat(catalog_ref, matchers.HasLength(1))
        self.assertThat(catalog_ref[0]['endpoints'], matchers.HasLength(1))
        self.assertEqual(endpoint_1['id'], catalog_ref[0]['endpoints'][0]['id'])

    def test_v3_catalog_endpoint_filter_disabled(self):
        self.config_fixture.config(group='endpoint_filter', return_all_endpoints_if_no_filter=True)
        srv_1 = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(srv_1['id'], srv_1)
        endpoint_1 = unit.new_endpoint_ref(service_id=srv_1['id'], region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint_1['id'], endpoint_1)
        srv_2 = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(srv_2['id'], srv_2)
        catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(uuid.uuid4().hex, self.project_bar['id'])
        self.assertThat(catalog_ref, matchers.HasLength(2))
        srv_id_list = [catalog_ref[0]['id'], catalog_ref[1]['id']]
        self.assertCountEqual([srv_1['id'], srv_2['id']], srv_id_list)