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
def _register_image_mocks(self):
    self.register_uris([dict(method='GET', uri='https://image.example.com/v2/images/{name}'.format(name=self.image_name), status_code=404), dict(method='GET', uri='https://image.example.com/v2/images?name={name}'.format(name=self.image_name), json=self.fake_search_return), dict(method='GET', uri='https://image.example.com/v2/images/{id}/file'.format(id=self.image_id), content=self.output, headers={'Content-Type': 'application/octet-stream', 'Content-MD5': self.fake_image_dict['checksum']})])