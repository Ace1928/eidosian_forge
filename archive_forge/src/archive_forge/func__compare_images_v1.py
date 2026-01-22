import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def _compare_images_v1(self, exp, real):
    self.assertDictEqual(image_v1.Image(**exp).to_dict(computed=False), real.to_dict(computed=False))