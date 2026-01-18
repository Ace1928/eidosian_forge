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
@mock.patch.object(db_api.utils, 'paginate_query')
def test_stack_get_all_filters_sort_keys(self, mock_paginate):
    sort_keys = ['stack_name', 'stack_status', 'creation_time', 'updated_time', 'stack_owner']
    db_api.stack_get_all(self.ctx, sort_keys=sort_keys)
    args = mock_paginate.call_args[0]
    used_sort_keys = set(args[3])
    expected_keys = set(['name', 'status', 'created_at', 'updated_at', 'id'])
    self.assertEqual(expected_keys, used_sort_keys)