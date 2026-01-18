import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base

        Test that rebuild_server with a wait returns the server instance when
        its status changes to "ACTIVE".
        