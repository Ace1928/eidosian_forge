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
def test_user1_finding_user1_memberships(self):
    """User1 should see all images shared with User1 """
    expected = [(self.owner1, 'shared-with-1'), (self.owner1, 'shared-with-both'), (self.owner2, 'shared-with-1'), (self.owner2, 'shared-with-both')]
    self._check_by_member(self.user1_ctx, self.tenant1, expected)