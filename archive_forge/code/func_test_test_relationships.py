import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_relationships(self):
    checker = fixture.ObjectVersionChecker()
    tree = checker.get_dependency_tree()
    actual = tree['TestSubclassedObject']
    tree['TestSubclassedObject']['Foo'] = '9.8'
    expected, actual = checker.test_relationships(tree)
    self.assertEqual(['TestSubclassedObject'], list(expected.keys()))
    self.assertEqual(['TestSubclassedObject'], list(actual.keys()))
    self.assertEqual({'MyOwnedObject': '1.0', 'Foo': '9.8'}, expected['TestSubclassedObject'])
    self.assertEqual({'MyOwnedObject': '1.0'}, actual['TestSubclassedObject'])