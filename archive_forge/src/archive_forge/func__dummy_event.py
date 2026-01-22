import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def _dummy_event(self, res_properties=None):
    resource = self.stack['generic1']
    ev_uuid = 'abc123yc-9f88-404d-a85b-531529456xyz'
    ev = event.Event(self.context, self.stack, 'CREATE', 'COMPLETE', 'state changed', 'z3455xyc-9f88-404d-a85b-5315293e67de', resource._rsrc_prop_data_id, resource._stored_properties_data, resource.name, resource.type(), uuid=ev_uuid)
    ev.store()
    return event_object.Event.get_all_by_stack(self.context, self.stack.id, filters={'uuid': ev_uuid})[0]