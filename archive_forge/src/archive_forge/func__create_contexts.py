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
def _create_contexts(self):
    self.owner1, self.owner1_ctx = self._user_fixture()
    self.owner2, self.owner2_ctx = self._user_fixture()
    self.tenant1, self.user1_ctx = self._user_fixture()
    self.tenant2, self.user2_ctx = self._user_fixture()
    self.tenant3, self.user3_ctx = self._user_fixture()
    self.admin_tenant, self.admin_ctx = self._user_fixture(admin=True)