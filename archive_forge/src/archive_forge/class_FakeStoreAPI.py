from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
class FakeStoreAPI(object):

    def __init__(self, store_metadata=None):
        self.data = {'%s/%s' % (BASE_URI, UUID1): ('XXX', 3), '%s/fake_location' % BASE_URI: ('YYY', 3)}
        self.acls = {}
        if store_metadata is None:
            self.store_metadata = {}
        else:
            self.store_metadata = store_metadata

    def create_stores(self):
        pass

    def set_acls(self, uri, public=False, read_tenants=None, write_tenants=None, context=None):
        if read_tenants is None:
            read_tenants = []
        if write_tenants is None:
            write_tenants = []
        self.acls[uri] = {'public': public, 'read': read_tenants, 'write': write_tenants}

    def get_from_backend(self, location, offset=0, chunk_size=None, context=None):
        try:
            scheme = location[:location.find('/') - 1]
            if scheme == 'unknown':
                raise store.UnknownScheme(scheme=scheme)
            return self.data[location]
        except KeyError:
            raise store.NotFound(image=location)

    def get_size_from_backend(self, location, context=None):
        return self.get_from_backend(location, context=context)[1]

    def add_to_backend(self, conf, image_id, data, size, scheme=None, context=None, verifier=None):
        store_max_size = 7
        current_store_size = 2
        for location in self.data.keys():
            if image_id in location:
                raise exception.Duplicate()
        if not size:
            size = len(data.data.fd)
        if current_store_size + size > store_max_size:
            raise exception.StorageFull()
        if context.user_id == USER2:
            raise exception.Forbidden()
        if context.user_id == USER3:
            raise exception.StorageWriteDenied()
        self.data[image_id] = (data, size)
        checksum = 'Z'
        return (image_id, size, checksum, self.store_metadata)

    def add_to_backend_with_multihash(self, conf, image_id, data, size, hashing_algo, scheme=None, context=None, verifier=None):
        store_max_size = 7
        current_store_size = 2
        for location in self.data.keys():
            if image_id in location:
                raise exception.Duplicate()
        if not size:
            size = len(data.data.fd)
        if current_store_size + size > store_max_size:
            raise exception.StorageFull()
        if context.user_id == USER2:
            raise exception.Forbidden()
        if context.user_id == USER3:
            raise exception.StorageWriteDenied()
        self.data[image_id] = (data, size)
        checksum = 'Z'
        multihash = 'ZZ'
        return (image_id, size, checksum, multihash, self.store_metadata)

    def check_location_metadata(self, val, key=''):
        store.check_location_metadata(val)

    def delete_from_backend(self, uri, context=None):
        pass