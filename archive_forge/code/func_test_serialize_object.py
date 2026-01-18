import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_serialize_object(self):
    self.assertEqual({}, vim_util.serialize_object({}))
    mobj1 = mock.MagicMock()
    mobj1.__keylist__ = ['asdf']
    mobj1.keys = lambda: ['asdf']
    mobj1.__getitem__.side_effect = [1]
    mobj2 = mock.Mock()
    mobj3 = mock.MagicMock()
    mobj3.__keylist__ = ['subkey1', 'subkey2']
    mobj3.keys = lambda: ['subkey1', 'subkey2']
    mobj3.__getitem__.side_effect = ['subvalue1', True]
    mobj4 = 12
    obj = {'foo': mobj1, 'bar': [mobj2, mobj3], 'baz': mobj4}
    expected = {'foo': {'asdf': 1}, 'bar': [mobj2, {'subkey1': 'subvalue1', 'subkey2': True}], 'baz': 12}
    self.assertEqual(expected, vim_util.serialize_object(obj))