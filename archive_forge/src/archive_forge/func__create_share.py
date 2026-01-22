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
def _create_share(self, stack_name):
    share = self._init_share(stack_name)
    share.client().shares.create.return_value = self.fake_share
    share.client().shares.get.return_value = self.available_share
    scheduler.TaskRunner(share.create)()
    return share