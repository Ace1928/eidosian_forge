import datetime
import os
from unittest import mock
import glance_store as store_api
from oslo_config import cfg
from glance.async_.flows._internal_plugins import copy_image
from glance.async_.flows import api_image_import
import glance.common.exception as exception
from glance import domain
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _create_images(self):
    self.images = [_db_fixture(UUID1, owner=TENANT1, checksum=CHKSUM, name='1', size=512, virtual_size=2048, visibility='public', disk_format='raw', container_format='bare', status='active', tags=['redhat', '64bit', 'power'], properties={'hypervisor_type': 'kvm', 'foo': 'bar', 'bar': 'foo'}, locations=[{'url': 'file://%s/%s' % (self.test_dir, UUID1), 'metadata': {'store': 'fast'}, 'status': 'active'}], created_at=DATETIME + datetime.timedelta(seconds=1))]
    [self.db.image_create(None, image) for image in self.images]
    self.db.image_tag_set_all(None, UUID1, ['ping', 'pong'])