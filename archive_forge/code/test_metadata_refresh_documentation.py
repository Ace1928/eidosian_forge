from unittest import mock
from oslo_serialization import jsonutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.cfn.wait_condition_handle import (
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack as stk
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
Tests a wait condition metadata update after a signal call.