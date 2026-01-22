from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
class TestStoreMetaDataChecker(utils.BaseTestCase):

    def test_empty(self):
        glance_store.check_location_metadata({})

    def test_unicode(self):
        m = {'key': 'somevalue'}
        glance_store.check_location_metadata(m)

    def test_unicode_list(self):
        m = {'key': ['somevalue', '2']}
        glance_store.check_location_metadata(m)

    def test_unicode_dict(self):
        inner = {'key1': 'somevalue', 'key2': 'somevalue'}
        m = {'topkey': inner}
        glance_store.check_location_metadata(m)

    def test_unicode_dict_list(self):
        inner = {'key1': 'somevalue', 'key2': 'somevalue'}
        m = {'topkey': inner, 'list': ['somevalue', '2'], 'u': '2'}
        glance_store.check_location_metadata(m)

    def test_nested_dict(self):
        inner = {'key1': 'somevalue', 'key2': 'somevalue'}
        inner = {'newkey': inner}
        inner = {'anotherkey': inner}
        m = {'topkey': inner}
        glance_store.check_location_metadata(m)

    def test_simple_bad(self):
        m = {'key1': object()}
        self.assertRaises(glance_store.BackendException, glance_store.check_location_metadata, m)

    def test_list_bad(self):
        m = {'key1': ['somevalue', object()]}
        self.assertRaises(glance_store.BackendException, glance_store.check_location_metadata, m)

    def test_nested_dict_bad(self):
        inner = {'key1': 'somevalue', 'key2': object()}
        inner = {'newkey': inner}
        inner = {'anotherkey': inner}
        m = {'topkey': inner}
        self.assertRaises(glance_store.BackendException, glance_store.check_location_metadata, m)