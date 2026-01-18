import io
import operator
import tempfile
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import _log
from openstack import exceptions
from openstack.image.v2 import image
from openstack.tests.unit import base
from openstack import utils
def test_stage_error(self):
    sot = image.Image(**EXAMPLE)
    self.sess.put.return_value = FakeResponse('dummy', status_code=400)
    self.assertRaises(exceptions.SDKException, sot.stage, self.sess)