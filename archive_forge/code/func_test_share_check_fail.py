import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_share_check_fail(self):
    share = self._create_share('stack_share_check_fail')
    share.client().shares.get.return_value = self.failed_share
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(share.check))
    self.assertIn("Error: resources.test_share: 'status': expected '['available']'", str(exc))