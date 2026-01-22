from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
class FakeEndpointData:
    min_microversion = '2.1'
    max_microversion = '2.78'