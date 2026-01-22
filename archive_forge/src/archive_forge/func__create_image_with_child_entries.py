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
def _create_image_with_child_entries():
    fixture = {'status': 'queued', 'locations': location_data}
    image_id = self.db_api.image_create(self.context, fixture)['id']
    fixture = {'name': 'ping', 'value': 'pong', 'image_id': image_id}
    self.db_api.image_property_create(self.context, fixture)
    fixture = {'image_id': image_id, 'member': TENANT2, 'can_share': False}
    self.db_api.image_member_create(self.context, fixture)
    self.db_api.image_tag_create(self.context, image_id, 'snarf')
    return image_id