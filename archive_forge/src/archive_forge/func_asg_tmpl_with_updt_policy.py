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
def asg_tmpl_with_updt_policy():
    t = template_format.parse(inline_templates.as_template)
    ag = t['Resources']['WebServerGroup']
    ag['UpdatePolicy'] = {'AutoScalingRollingUpdate': {'MinInstancesInService': '1', 'MaxBatchSize': '2', 'PauseTime': 'PT1S'}}
    return json.dumps(t)