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
class TestMembershipVisibility(test_utils.BaseTestCase):

    def setUp(self):
        super(TestMembershipVisibility, self).setUp()
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)
        self._create_contexts()
        self._create_images()

    def _create_contexts(self):
        self.owner1, self.owner1_ctx = self._user_fixture()
        self.owner2, self.owner2_ctx = self._user_fixture()
        self.tenant1, self.user1_ctx = self._user_fixture()
        self.tenant2, self.user2_ctx = self._user_fixture()
        self.tenant3, self.user3_ctx = self._user_fixture()
        self.admin_tenant, self.admin_ctx = self._user_fixture(admin=True)

    def _user_fixture(self, admin=False):
        tenant_id = str(uuid.uuid4())
        ctx = context.RequestContext(tenant=tenant_id, is_admin=admin)
        return (tenant_id, ctx)

    def _create_images(self):
        self.image_ids = {}
        for owner in [self.owner1, self.owner2]:
            self._create_image('not_shared', owner)
            self._create_image('shared-with-1', owner, members=[self.tenant1])
            self._create_image('shared-with-2', owner, members=[self.tenant2])
            self._create_image('shared-with-both', owner, members=[self.tenant1, self.tenant2])

    def _create_image(self, name, owner, members=None):
        image = build_image_fixture(name=name, owner=owner, visibility='shared')
        self.image_ids[owner, name] = image['id']
        self.db_api.image_create(self.admin_ctx, image)
        for member in members or []:
            member = {'image_id': image['id'], 'member': member}
            self.db_api.image_member_create(self.admin_ctx, member)