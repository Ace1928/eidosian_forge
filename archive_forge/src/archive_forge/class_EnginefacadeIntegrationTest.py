import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
class EnginefacadeIntegrationTest(test_base.BaseTestCase):

    def test_db_fixture(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite://', sqlite_fk=True, mysql_sql_mode='FOOBAR', max_overflow=38)

        class MyFixture(test_fixtures.OpportunisticDbFixture):

            def get_enginefacade(self):
                return normal_mgr
        test = mock.Mock(SCHEMA_SCOPE=None)
        fixture = MyFixture(test=test)
        resources = fixture._get_resources()
        testresources.setUpResources(test, resources, None)
        self.addCleanup(testresources.tearDownResources, test, resources, None)
        fixture.setUp()
        self.addCleanup(fixture.cleanUp)
        self.assertTrue(normal_mgr._factory._started)
        test.engine = normal_mgr.writer.get_engine()
        self.assertEqual('sqlite://', str(test.engine.url))
        self.assertIs(test.engine, normal_mgr._factory._writer_engine)
        engine_args = normal_mgr._factory._engine_args_for_conf(None)
        self.assertTrue(engine_args['sqlite_fk'])
        self.assertEqual('FOOBAR', engine_args['mysql_sql_mode'])
        self.assertEqual(38, engine_args['max_overflow'])
        fixture.cleanUp()
        fixture._clear_cleanups()
        self.assertFalse(normal_mgr._factory._started)