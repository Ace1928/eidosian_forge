import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
This test verifies that when list_servers is called with
        `filters` dict that it passes it to nova.