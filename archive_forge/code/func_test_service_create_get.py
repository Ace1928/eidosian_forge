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