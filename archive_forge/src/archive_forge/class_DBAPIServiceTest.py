import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class DBAPIServiceTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIServiceTest, self).setUp()
        self.ctx = utils.dummy_context()

    def test_service_create_get(self):
        service = create_service(self.ctx)
        ret_service = db_api.service_get(self.ctx, service.id)
        self.assertIsNotNone(ret_service)
        self.assertEqual(service.id, ret_service.id)
        self.assertEqual(service.hostname, ret_service.hostname)
        self.assertEqual(service.binary, ret_service.binary)
        self.assertEqual(service.host, ret_service.host)
        self.assertEqual(service.topic, ret_service.topic)
        self.assertEqual(service.engine_id, ret_service.engine_id)
        self.assertEqual(service.report_interval, ret_service.report_interval)
        self.assertIsNotNone(service.created_at)
        self.assertIsNone(service.updated_at)
        self.assertIsNone(service.deleted_at)

    def test_service_get_all_by_args(self):
        values = [{'id': str(uuid.uuid4()), 'hostname': 'host-1', 'host': 'engine-1'}]
        for i in [0, 1, 2]:
            values.append({'id': str(uuid.uuid4()), 'hostname': 'host-2', 'host': 'engine-%s' % i})
        [create_service(self.ctx, **val) for val in values]
        services = db_api.service_get_all(self.ctx)
        self.assertEqual(4, len(services))
        services_by_args = db_api.service_get_all_by_args(self.ctx, hostname='host-2', binary='heat-engine', host='engine-0')
        self.assertEqual(1, len(services_by_args))
        self.assertEqual('host-2', services_by_args[0].hostname)
        self.assertEqual('heat-engine', services_by_args[0].binary)
        self.assertEqual('engine-0', services_by_args[0].host)

    def test_service_update(self):
        service = create_service(self.ctx)
        values = {'hostname': 'host-updated', 'host': 'engine-updated', 'retry_interval': 120}
        service = db_api.service_update(self.ctx, service.id, values)
        self.assertEqual('host-updated', service.hostname)
        self.assertEqual(120, service.retry_interval)
        self.assertEqual('engine-updated', service.host)
        old_updated_date = service.updated_at
        service = db_api.service_update(self.ctx, service.id, dict())
        self.assertGreater(service.updated_at, old_updated_date)

    def test_service_delete_soft_delete(self):
        service = create_service(self.ctx)
        db_api.service_delete(self.ctx, service.id)
        ret_service = db_api.service_get(self.ctx, service.id)
        self.assertEqual(ret_service.id, service.id)
        db_api.service_delete(self.ctx, service.id, False)
        ex = self.assertRaises(exception.EntityNotFound, db_api.service_get, self.ctx, service.id)
        self.assertEqual('Service', ex.kwargs.get('entity'))