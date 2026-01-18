from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.events as events
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
Tests the API class EventController.

    Tests the API class which acts as the WSGI controller,
    the endpoint processing API requests after they are routed
    