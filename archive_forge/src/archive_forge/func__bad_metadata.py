from unittest import mock
from glance_store import backend
from glance_store import exceptions
from glance_store.tests import base
def _bad_metadata(self, in_metadata):
    mstore = mock.Mock()
    mstore.add.return_value = (self.location, self.size, self.checksum, in_metadata)
    mstore.__str__ = lambda self: 'hello'
    mstore.__unicode__ = lambda self: 'hello'
    self.assertRaises(exceptions.BackendException, backend.store_add_to_backend, self.image_id, self.data, self.size, mstore)
    mstore.add.assert_called_once_with(self.image_id, mock.ANY, self.size, context=None, verifier=None)
    newstore = mock.Mock()
    newstore.add.return_value = (self.location, self.size, self.checksum, self.multihash, in_metadata)
    newstore.__str__ = lambda self: 'hello'
    newstore.__unicode__ = lambda self: 'hello'
    self.assertRaises(exceptions.BackendException, backend.store_add_to_backend_with_multihash, self.image_id, self.data, self.size, self.hash_algo, newstore)
    newstore.add.assert_called_once_with(self.image_id, mock.ANY, self.size, self.hash_algo, context=None, verifier=None)