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
def test_paginate_query_gets_model_marker(self, mock_paginate_query):
    query = mock.Mock()
    model = mock.Mock()
    marker = mock.Mock()
    result = 'real_marker'
    ctx = mock.MagicMock()
    ctx.session.get.return_value = result
    db_api._paginate_query(ctx, query, model, marker=marker)
    ctx.session.get.assert_called_once_with(model, marker)
    args, _ = mock_paginate_query.call_args
    self.assertIn(result, args)