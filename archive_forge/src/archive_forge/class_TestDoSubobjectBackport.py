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
class TestDoSubobjectBackport(test.TestCase):

    @base.VersionedObjectRegistry.register
    class ParentObj(base.VersionedObject):
        VERSION = '1.1'
        fields = {'child': fields.ObjectField('ChildObj', nullable=True)}
        obj_relationships = {'child': [('1.0', '1.0'), ('1.1', '1.1')]}

    @base.VersionedObjectRegistry.register
    class ParentObjList(base.VersionedObject, base.ObjectListBase):
        VERSION = '1.1'
        fields = {'objects': fields.ListOfObjectsField('ChildObj')}
        obj_relationships = {'objects': [('1.0', '1.0'), ('1.1', '1.1')]}

    @base.VersionedObjectRegistry.register
    class ChildObj(base.VersionedObject):
        VERSION = '1.1'
        fields = {'foo': fields.IntegerField()}

    def test_do_subobject_backport_without_manifest(self):
        child = self.ChildObj(foo=1)
        parent = self.ParentObj(child=child)
        parent_primitive = parent.obj_to_primitive()['versioned_object.data']
        primitive = child.obj_to_primitive()['versioned_object.data']
        version = '1.0'
        compat_func = 'obj_make_compatible_from_manifest'
        with mock.patch.object(child, compat_func) as mock_compat:
            base._do_subobject_backport(version, parent, 'child', parent_primitive)
            mock_compat.assert_called_once_with(primitive, version, version_manifest=None)

    def test_do_subobject_backport_with_manifest(self):
        child = self.ChildObj(foo=1)
        parent = self.ParentObj(child=child)
        parent_primitive = parent.obj_to_primitive()['versioned_object.data']
        primitive = child.obj_to_primitive()['versioned_object.data']
        version = '1.0'
        manifest = {'ChildObj': '1.0'}
        parent._obj_version_manifest = manifest
        compat_func = 'obj_make_compatible_from_manifest'
        with mock.patch.object(child, compat_func) as mock_compat:
            base._do_subobject_backport(version, parent, 'child', parent_primitive)
            mock_compat.assert_called_once_with(primitive, version, version_manifest=manifest)

    def test_do_subobject_backport_with_manifest_old_parent(self):
        child = self.ChildObj(foo=1)
        parent = self.ParentObj(child=child)
        manifest = {'ChildObj': '1.0'}
        parent_primitive = parent.obj_to_primitive(target_version='1.1', version_manifest=manifest)
        child_primitive = parent_primitive['versioned_object.data']['child']
        self.assertEqual('1.0', child_primitive['versioned_object.version'])

    def test_do_subobject_backport_list_object(self):
        child = self.ChildObj(foo=1)
        parent = self.ParentObjList(objects=[child])
        parent_primitive = parent.obj_to_primitive()['versioned_object.data']
        primitive = child.obj_to_primitive()['versioned_object.data']
        version = '1.0'
        compat_func = 'obj_make_compatible_from_manifest'
        with mock.patch.object(child, compat_func) as mock_compat:
            base._do_subobject_backport(version, parent, 'objects', parent_primitive)
            mock_compat.assert_called_once_with(primitive, version, version_manifest=None)

    def test_do_subobject_backport_list_object_with_manifest(self):
        child = self.ChildObj(foo=1)
        parent = self.ParentObjList(objects=[child])
        manifest = {'ChildObj': '1.0', 'ParentObjList': '1.0'}
        parent_primitive = parent.obj_to_primitive(target_version='1.0', version_manifest=manifest)
        self.assertEqual('1.0', parent_primitive['versioned_object.version'])
        child_primitive = parent_primitive['versioned_object.data']['objects']
        self.assertEqual('1.0', child_primitive[0]['versioned_object.version'])

    def test_do_subobject_backport_null_child(self):
        parent = self.ParentObj(child=None)
        parent_primitive = parent.obj_to_primitive()['versioned_object.data']
        version = '1.0'
        compat_func = 'obj_make_compatible_from_manifest'
        with mock.patch.object(self.ChildObj, compat_func) as mock_compat:
            base._do_subobject_backport(version, parent, 'child', parent_primitive)
            self.assertFalse(mock_compat.called, 'obj_make_compatible_from_manifest() should not have been called because the subobject is None.')

    def test_to_primitive_calls_make_compatible_manifest(self):
        obj = self.ParentObj()
        with mock.patch.object(obj, 'obj_make_compatible_from_manifest') as m:
            obj.obj_to_primitive(target_version='1.0', version_manifest=mock.sentinel.manifest)
            m.assert_called_once_with(mock.ANY, '1.0', mock.sentinel.manifest)