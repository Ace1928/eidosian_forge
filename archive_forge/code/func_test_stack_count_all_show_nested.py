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
def test_stack_count_all_show_nested(self):
    stack1 = self._setup_test_stack('stack1', UUID1)[1]
    self._setup_test_stack('stack2', UUID2, owner_id=stack1.id)
    self._setup_test_stack('stack1*', UUID3, owner_id=stack1.id, backup=True)
    st_db = db_api.stack_count_all(self.ctx)
    self.assertEqual(1, st_db)
    st_db = db_api.stack_count_all(self.ctx, show_nested=True)
    self.assertEqual(2, st_db)