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
def _call_create_image(self, name, **kwargs):
    imagefile = tempfile.NamedTemporaryFile(delete=False)
    imagefile.write(b'\x00')
    imagefile.close()
    self.cloud.create_image(name, imagefile.name, wait=True, timeout=1, is_public=False, validate_checksum=True, **kwargs)