import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_scaling_not_in_progress_legacy(self):
    awhile_ago = timeutils.utcnow() - datetime.timedelta(seconds=100)
    previous_meta = {'cooldown': {awhile_ago.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    self.assertIsNone(self.group._check_scaling_allowed(60))