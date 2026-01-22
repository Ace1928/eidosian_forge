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
class DBPurgeTests(test_utils.BaseTestCase):

    def setUp(self):
        super(DBPurgeTests, self).setUp()
        self.adm_context = context.get_admin_context(show_deleted=True)
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)
        self.context = context.RequestContext(is_admin=True)
        self.image_fixtures, self.task_fixtures = self.build_fixtures()
        self.create_tasks(self.task_fixtures)
        self.create_images(self.image_fixtures)

    def build_fixtures(self):
        dt1 = timeutils.utcnow() - datetime.timedelta(days=5)
        dt2 = dt1 + datetime.timedelta(days=1)
        dt3 = dt2 + datetime.timedelta(days=1)
        fixtures = [{'created_at': dt1, 'updated_at': dt1, 'deleted_at': dt3, 'deleted': True}, {'created_at': dt1, 'updated_at': dt2, 'deleted_at': timeutils.utcnow(), 'deleted': True}, {'created_at': dt2, 'updated_at': dt2, 'deleted_at': None, 'deleted': False}]
        return ([build_image_fixture(**fixture) for fixture in fixtures], [build_task_fixture(**fixture) for fixture in fixtures])

    def create_images(self, images):
        for fixture in images:
            self.db_api.image_create(self.adm_context, fixture)

    def create_tasks(self, tasks):
        for fixture in tasks:
            self.db_api.task_create(self.adm_context, fixture)

    def test_db_purge(self):
        self.db_api.purge_deleted_rows(self.adm_context, 1, 5)
        images = self.db_api.image_get_all(self.adm_context)
        self.assertEqual(len(images), 3)
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(len(tasks), 2)

    def test_db_purge_images_table(self):
        images = self.db_api.image_get_all(self.adm_context)
        self.assertEqual(len(images), 3)
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(len(tasks), 3)
        for image in images:
            session = self.db_api.get_session()
            with session.begin():
                session.execute(sql.delete(models.ImageLocation).where(models.ImageLocation.image_id == image['id']))
        self.db_api.purge_deleted_rows(self.adm_context, 1, 5)
        self.db_api.purge_deleted_rows_from_images(self.adm_context, 1, 5)
        images = self.db_api.image_get_all(self.adm_context)
        self.assertEqual(len(images), 2)
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(len(tasks), 2)

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

    def test_purge_task_info_with_refs_to_soft_deleted_tasks(self):
        session = db_api.get_session()
        engine = db_api.get_engine()
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(3, len(tasks))
        task_info = sqlalchemyutils.get_table(engine, 'task_info')
        with session.begin():
            task_info_rows = session.query(task_info).count()
        self.assertEqual(3, task_info_rows)
        self.db_api.purge_deleted_rows(self.context, 1, 5)
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(2, len(tasks))
        with session.begin():
            task_info_rows = session.query(task_info).count()
        self.assertEqual(2, task_info_rows)