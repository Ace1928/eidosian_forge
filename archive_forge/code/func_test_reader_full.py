import hashlib
import io
from unittest import mock
import uuid
from oslo_utils import secretutils
from oslo_utils import units
from oslo_vmware import api
from oslo_vmware import exceptions as vmware_exceptions
from oslo_vmware.objects import datacenter as oslo_datacenter
from oslo_vmware.objects import datastore as oslo_datastore
import glance_store._drivers.vmware_datastore as vm_store
from glance_store import backend
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_reader_full(self):
    content = b'XXX'
    image = io.BytesIO(content)
    expected_checksum = secretutils.md5(content, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(content).hexdigest()
    reader = vm_store._Reader(image, self.hash_algo)
    ret = reader.read()
    self.assertEqual(content, ret)
    self.assertEqual(expected_checksum, reader.checksum.hexdigest())
    self.assertEqual(expected_multihash, reader.os_hash_value.hexdigest())
    self.assertEqual(len(content), reader.size)