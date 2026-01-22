import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class TestVisibility(test_utils.BaseTestCase):

    def setUp(self):
        super(TestVisibility, self).setUp()
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)
        self.setup_tenants()
        self.setup_contexts()
        self.fixtures = self.build_image_fixtures()
        self.create_images(self.fixtures)

    def setup_tenants(self):
        self.admin_tenant = str(uuid.uuid4())
        self.tenant1 = str(uuid.uuid4())
        self.tenant2 = str(uuid.uuid4())

    def setup_contexts(self):
        self.admin_context = context.RequestContext(is_admin=True, tenant=self.admin_tenant)
        self.admin_none_context = context.RequestContext(is_admin=True, tenant=None)
        self.tenant1_context = context.RequestContext(tenant=self.tenant1)
        self.tenant2_context = context.RequestContext(tenant=self.tenant2)
        self.none_context = context.RequestContext(tenant=None)

    def build_image_fixtures(self):
        fixtures = []
        owners = {'Unowned': None, 'Admin Tenant': self.admin_tenant, 'Tenant 1': self.tenant1, 'Tenant 2': self.tenant2}
        visibilities = ['community', 'private', 'public', 'shared']
        for owner_label, owner in owners.items():
            for visibility in visibilities:
                fixture = {'name': '%s, %s' % (owner_label, visibility), 'owner': owner, 'visibility': visibility}
                fixtures.append(fixture)
        return [build_image_fixture(**f) for f in fixtures]

    def create_images(self, images):
        for fixture in images:
            self.db_api.image_create(self.admin_context, fixture)