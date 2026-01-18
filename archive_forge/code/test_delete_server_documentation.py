import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base

        Test that deleting server with a borked neutron doesn't bork
        