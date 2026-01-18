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
def test_obj_set_field_modifications(self):

    @base.VersionedObjectRegistry.register
    class ObjWithSet(base.VersionedObject):
        fields = {'set_field': fields.Field(fields.Set(fields.Integer()))}
    obj = ObjWithSet()
    obj.set_field = set([42])

    def add(value):
        obj.set_field.add(value)

    def update_w_set(value):
        obj.set_field.update(set([value]))

    def update_w_list(value):
        obj.set_field.update([value, value, value])

    def sym_diff_upd(value):
        obj.set_field.symmetric_difference_update(set([value]))

    def union(value):
        obj.set_field = obj.set_field | set([value])

    def iunion(value):
        obj.set_field |= set([value])

    def xor(value):
        obj.set_field = obj.set_field ^ set([value])

    def ixor(value):
        obj.set_field ^= set([value])
    sym_diff_upd('42')
    add('1')
    update_w_list('2')
    update_w_set('3')
    union('4')
    iunion('5')
    xor('6')
    ixor('7')
    self.assertEqual(set([1, 2, 3, 4, 5, 6, 7]), obj.set_field)
    obj.set_field = set([42])
    obj.obj_reset_changes()
    self.assertRaises(ValueError, add, 'abc')
    self.assertRaises(ValueError, update_w_list, 'abc')
    self.assertRaises(ValueError, update_w_set, 'abc')
    self.assertRaises(ValueError, sym_diff_upd, 'abc')
    self.assertRaises(ValueError, union, 'abc')
    self.assertRaises(ValueError, iunion, 'abc')
    self.assertRaises(ValueError, xor, 'abc')
    self.assertRaises(ValueError, ixor, 'abc')
    self.assertEqual(set([42]), obj.set_field)
    self.assertEqual({}, obj.obj_get_changes())