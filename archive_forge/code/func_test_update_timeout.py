import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_update_timeout(self):
    self.stack.timeout_secs = mock.Mock(return_value=100)
    self.assertEqual(60, self.instance_group._update_timeout(batch_cnt=3, pause_sec=20))