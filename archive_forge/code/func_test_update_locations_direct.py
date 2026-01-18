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
def test_update_locations_direct(self):
    """
        For some reasons update_locations can be called directly
        (not via image_update), so better check that everything is ok if passed
        4 byte unicode characters
        """
    location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}]
    fixture = {'locations': location_data}
    image = self.db_api.image_update(self.adm_context, UUID1, fixture)
    self.assertEqual(1, len(image['locations']))
    self.assertIn('id', image['locations'][0])
    loc_id = image['locations'][0].pop('id')
    bad_location = {'url': 'Bad 😊', 'metadata': {}, 'status': 'active', 'id': loc_id}
    self.assertRaises(exception.Invalid, self.db_api.image_location_update, self.adm_context, UUID1, bad_location)