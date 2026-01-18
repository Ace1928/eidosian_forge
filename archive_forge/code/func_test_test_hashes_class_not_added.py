import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_hashes_class_not_added(self):
    fp = 'gyrados'
    new_classes = copy.copy(self.obj_classes)
    self._add_class(new_classes, MyExtraObject)
    expected_hashes = self._generate_hashes(self.obj_classes, fp)
    actual_hashes = self._generate_hashes(new_classes, fp)
    with mock.patch.object(self.ovc, 'get_hashes') as mock_gh:
        mock_gh.return_value = actual_hashes
        actual_exp, actual_act = self.ovc.test_hashes(expected_hashes)
    expected_expected = {MyExtraObject.__name__: None}
    expected_actual = {MyExtraObject.__name__: fp}
    self.assertEqual(expected_expected, actual_exp, 'Expected hashes should not contain the fingerprint of the class that has not been added to the expected hash dictionary.')
    self.assertEqual(expected_actual, actual_act, 'The actual hash should contain the class that was added to the registry.')