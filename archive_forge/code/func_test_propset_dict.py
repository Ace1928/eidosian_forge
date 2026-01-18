import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_propset_dict(self):
    self.assertEqual({}, vim_util.propset_dict(None))
    mock_propset = []
    for i in range(2):
        mock_obj = mock.Mock()
        mock_obj.name = 'test_name_%d' % i
        mock_obj.val = 'test_val_%d' % i
        mock_propset.append(mock_obj)
    self.assertEqual({'test_name_0': 'test_val_0', 'test_name_1': 'test_val_1'}, vim_util.propset_dict(mock_propset))