import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def _compare_containers(self, exp, real):
    self.assertDictEqual(container.Container(**exp).to_dict(computed=False), real.to_dict(computed=False))