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
def test_purge_images_table_fk_constraint_failure(self):
    """Test foreign key constraint failure

        Test whether foreign key constraint failure during purge
        operation is raising DBReferenceError or not.
        """
    session = db_api.get_session()
    engine = db_api.get_engine()
    connection = engine.connect()
    images = sqlalchemyutils.get_table(engine, 'images')
    image_tags = sqlalchemyutils.get_table(engine, 'image_tags')
    uuidstr = uuid.uuid4().hex
    created_time = timeutils.utcnow() - datetime.timedelta(days=20)
    deleted_time = created_time + datetime.timedelta(days=5)
    images_row_fixture = {'id': uuidstr, 'status': 'status', 'created_at': created_time, 'deleted_at': deleted_time, 'deleted': 1, 'visibility': 'public', 'min_disk': 1, 'min_ram': 1, 'protected': 0}
    ins_stmt = images.insert().values(**images_row_fixture)
    with connection.begin():
        connection.execute(ins_stmt)
    image_tags_row_fixture = {'image_id': uuidstr, 'value': 'tag_value', 'created_at': created_time, 'deleted': 0}
    ins_stmt = image_tags.insert().values(**image_tags_row_fixture)
    with connection.begin():
        connection.execute(ins_stmt)
    self.assertRaises(db_exception.DBReferenceError, db_api.purge_deleted_rows_from_images, self.adm_context, age_in_days=10, max_rows=50)
    with session.begin():
        images_rows = session.query(images).count()
    self.assertEqual(4, images_rows)