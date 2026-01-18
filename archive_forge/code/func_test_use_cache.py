import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_use_cache(self):
    cache = extension.ExtensionManager.ENTRY_POINT_CACHE
    cache['stevedore.test.faux'] = []
    with mock.patch('stevedore._cache.get_group_all', side_effect=AssertionError('called get_group_all')):
        em = extension.ExtensionManager('stevedore.test.faux')
        names = em.names()
    self.assertEqual(names, [])