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
@mock.patch.object(vm_store.Store, '_get_datastore')
@mock.patch.object(vm_store.Store, '_get_freespace')
def test_select_datastore_equal_freespace(self, mock_get_freespace, mock_ds_obj):
    datastores = ['a:b:100', 'c:d:100', 'e:f:200']
    image_size = 10
    mock_ds_obj.side_effect = fake_datastore_obj
    self.store.datastores = self.store._build_datastore_weighted_map(datastores)
    freespaces = [11, 11, 11]

    def fake_get_fp(*args, **kwargs):
        return freespaces.pop(0)
    mock_get_freespace.side_effect = fake_get_fp
    ds = self.store.select_datastore(image_size)
    self.assertEqual('e', ds.datacenter.path)
    self.assertEqual('f', ds.name)