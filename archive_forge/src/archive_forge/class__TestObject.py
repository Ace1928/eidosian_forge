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
class _TestObject(object):

    def test_hydration_type_error(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.5', 'versioned_object.data': {'foo': 'a'}}
        self.assertRaises(ValueError, MyObj.obj_from_primitive, primitive)

    def test_hydration(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.5', 'versioned_object.data': {'foo': 1}}
        real_method = MyObj._obj_from_primitive

        def _obj_from_primitive(*args):
            return real_method(*args)
        with mock.patch.object(MyObj, '_obj_from_primitive') as ofp:
            ofp.side_effect = _obj_from_primitive
            obj = MyObj.obj_from_primitive(primitive)
            ofp.assert_called_once_with(None, '1.5', primitive)
        self.assertEqual(obj.foo, 1)

    def test_hydration_version_different(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.2', 'versioned_object.data': {'foo': 1}}
        obj = MyObj.obj_from_primitive(primitive)
        self.assertEqual(obj.foo, 1)
        self.assertEqual('1.2', obj.VERSION)

    def test_hydration_bad_ns(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'foo', 'versioned_object.version': '1.5', 'versioned_object.data': {'foo': 1}}
        self.assertRaises(exception.UnsupportedObjectError, MyObj.obj_from_primitive, primitive)

    def test_hydration_additional_unexpected_stuff(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.5.1', 'versioned_object.data': {'foo': 1, 'unexpected_thing': 'foobar'}}
        obj = MyObj.obj_from_primitive(primitive)
        self.assertEqual(1, obj.foo)
        self.assertFalse(hasattr(obj, 'unexpected_thing'))
        self.assertEqual('1.5.1', obj.VERSION)

    def test_dehydration(self):
        expected = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6', 'versioned_object.data': {'foo': 1}}
        obj = MyObj(foo=1)
        obj.obj_reset_changes()
        self.assertEqual(obj.obj_to_primitive(), expected)

    def test_dehydration_invalid_version(self):
        obj = MyObj(foo=1)
        obj.obj_reset_changes()
        self.assertRaises(exception.InvalidTargetVersion, obj.obj_to_primitive, target_version='1.7')

    def test_dehydration_same_version(self):
        expected = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6', 'versioned_object.data': {'foo': 1}}
        obj = MyObj(foo=1)
        obj.obj_reset_changes()
        with mock.patch.object(obj, 'obj_make_compatible') as mock_compat:
            self.assertEqual(obj.obj_to_primitive(target_version='1.6'), expected)
            self.assertFalse(mock_compat.called)

    def test_object_property(self):
        obj = MyObj(foo=1)
        self.assertEqual(obj.foo, 1)

    def test_object_property_type_error(self):
        obj = MyObj()

        def fail():
            obj.foo = 'a'
        self.assertRaises(ValueError, fail)

    def test_object_dict_syntax(self):
        obj = MyObj(foo=123, bar='text')
        self.assertEqual(obj['foo'], 123)
        self.assertIn('bar', obj)
        self.assertNotIn('missing', obj)
        self.assertEqual(sorted(iter(obj)), ['bar', 'foo'])
        self.assertEqual(sorted(obj.keys()), ['bar', 'foo'])
        self.assertEqual(sorted(obj.values(), key=str), [123, 'text'])
        self.assertEqual(sorted(obj.items()), [('bar', 'text'), ('foo', 123)])
        self.assertEqual(dict(obj), {'foo': 123, 'bar': 'text'})

    def test_non_dict_remotable(self):

        @base.VersionedObjectRegistry.register
        class TestObject(base.VersionedObject):

            @base.remotable
            def test_method(self):
                return 123
        obj = TestObject(context=self.context)
        self.assertEqual(123, obj.test_method())

    def test_load(self):
        obj = MyObj()
        self.assertEqual(obj.bar, 'loaded!')

    def test_load_in_base(self):

        @base.VersionedObjectRegistry.register
        class Foo(base.VersionedObject):
            fields = {'foobar': fields.Field(fields.Integer())}
        obj = Foo()
        with self.assertRaisesRegex(NotImplementedError, '.*foobar.*'):
            obj.foobar

    def test_loaded_in_primitive(self):
        obj = MyObj(foo=1)
        obj.obj_reset_changes()
        self.assertEqual(obj.bar, 'loaded!')
        expected = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6', 'versioned_object.changes': ['bar'], 'versioned_object.data': {'foo': 1, 'bar': 'loaded!'}}
        self.assertEqual(obj.obj_to_primitive(), expected)

    def test_changes_in_primitive(self):
        obj = MyObj(foo=123)
        self.assertEqual(obj.obj_what_changed(), set(['foo']))
        primitive = obj.obj_to_primitive()
        self.assertIn('versioned_object.changes', primitive)
        obj2 = MyObj.obj_from_primitive(primitive)
        self.assertEqual(obj2.obj_what_changed(), set(['foo']))
        obj2.obj_reset_changes()
        self.assertEqual(obj2.obj_what_changed(), set())

    def test_obj_class_from_name(self):
        obj = base.VersionedObject.obj_class_from_name('MyObj', '1.5')
        self.assertEqual('1.5', obj.VERSION)

    def test_obj_class_from_name_latest_compatible(self):
        obj = base.VersionedObject.obj_class_from_name('MyObj', '1.1')
        self.assertEqual('1.6', obj.VERSION)

    def test_unknown_objtype(self):
        self.assertRaises(exception.UnsupportedObjectError, base.VersionedObject.obj_class_from_name, 'foo', '1.0')

    def test_obj_class_from_name_supported_version(self):
        self.assertRaises(exception.IncompatibleObjectVersion, base.VersionedObject.obj_class_from_name, 'MyObj', '1.25')
        try:
            base.VersionedObject.obj_class_from_name('MyObj', '1.25')
        except exception.IncompatibleObjectVersion as error:
            self.assertEqual('1.6', error.kwargs['supported'])

    def test_orphaned_object(self):
        obj = MyObj.query(self.context)
        obj._context = None
        self.assertRaises(exception.OrphanedObjectError, obj._update_test)

    def test_changed_1(self):
        obj = MyObj.query(self.context)
        obj.foo = 123
        self.assertEqual(obj.obj_what_changed(), set(['foo']))
        obj._update_test()
        self.assertEqual(obj.obj_what_changed(), set(['foo', 'bar']))
        self.assertEqual(obj.foo, 123)

    def test_changed_2(self):
        obj = MyObj.query(self.context)
        obj.foo = 123
        self.assertEqual(obj.obj_what_changed(), set(['foo']))
        obj.save()
        self.assertEqual(obj.obj_what_changed(), set([]))
        self.assertEqual(obj.foo, 123)

    def test_changed_3(self):
        obj = MyObj.query(self.context)
        obj.foo = 123
        self.assertEqual(obj.obj_what_changed(), set(['foo']))
        obj.refresh()
        self.assertEqual(obj.obj_what_changed(), set([]))
        self.assertEqual(obj.foo, 321)
        self.assertEqual(obj.bar, 'refreshed')

    def test_changed_4(self):
        obj = MyObj.query(self.context)
        obj.bar = 'something'
        self.assertEqual(obj.obj_what_changed(), set(['bar']))
        obj.modify_save_modify()
        self.assertEqual(obj.obj_what_changed(), set(['foo', 'rel_object']))
        self.assertEqual(obj.foo, 42)
        self.assertEqual(obj.bar, 'meow')
        self.assertIsInstance(obj.rel_object, MyOwnedObject)

    def test_changed_with_sub_object(self):

        @base.VersionedObjectRegistry.register
        class ParentObject(base.VersionedObject):
            fields = {'foo': fields.IntegerField(), 'bar': fields.ObjectField('MyObj')}
        obj = ParentObject()
        self.assertEqual(set(), obj.obj_what_changed())
        obj.foo = 1
        self.assertEqual(set(['foo']), obj.obj_what_changed())
        bar = MyObj()
        obj.bar = bar
        self.assertEqual(set(['foo', 'bar']), obj.obj_what_changed())
        obj.obj_reset_changes()
        self.assertEqual(set(), obj.obj_what_changed())
        bar.foo = 1
        self.assertEqual(set(['bar']), obj.obj_what_changed())

    def test_changed_with_bogus_field(self):
        obj = MyObj()
        obj.foo = 123
        obj._changed_fields.add('does_not_exist')
        self.assertEqual(set(['foo']), obj.obj_what_changed())
        self.assertEqual({'foo': 123}, obj.obj_get_changes())

    def test_static_result(self):
        obj = MyObj.query(self.context)
        self.assertEqual(obj.bar, 'bar')
        result = obj.marco()
        self.assertEqual(result, 'polo')

    def test_updates(self):
        obj = MyObj.query(self.context)
        self.assertEqual(obj.foo, 1)
        obj._update_test()
        self.assertEqual(obj.bar, 'updated')

    def test_contains(self):
        obj = MyOwnedObject()
        self.assertNotIn('baz', obj)
        obj.baz = 1
        self.assertIn('baz', obj)
        self.assertNotIn('does_not_exist', obj)

    def test_obj_attr_is_set(self):
        obj = MyObj(foo=1)
        self.assertTrue(obj.obj_attr_is_set('foo'))
        self.assertFalse(obj.obj_attr_is_set('bar'))
        self.assertRaises(AttributeError, obj.obj_attr_is_set, 'bang')

    def test_obj_reset_changes_recursive(self):
        obj = MyObj(rel_object=MyOwnedObject(baz=123), rel_objects=[MyOwnedObject(baz=456)])
        self.assertEqual(set(['rel_object', 'rel_objects']), obj.obj_what_changed())
        obj.obj_reset_changes()
        self.assertEqual(set(['rel_object']), obj.obj_what_changed())
        self.assertEqual(set(['baz']), obj.rel_object.obj_what_changed())
        self.assertEqual(set(['baz']), obj.rel_objects[0].obj_what_changed())
        obj.obj_reset_changes(recursive=True, fields=['foo'])
        self.assertEqual(set(['rel_object']), obj.obj_what_changed())
        self.assertEqual(set(['baz']), obj.rel_object.obj_what_changed())
        self.assertEqual(set(['baz']), obj.rel_objects[0].obj_what_changed())
        obj.obj_reset_changes(recursive=True)
        self.assertEqual(set([]), obj.rel_object.obj_what_changed())
        self.assertEqual(set([]), obj.obj_what_changed())

    def test_get(self):
        obj = MyObj(foo=1)
        self.assertEqual(obj.get('foo', 2), 1)
        self.assertEqual(obj.get('foo'), 1)
        self.assertEqual(obj.get('bar', 'not-loaded'), 'not-loaded')
        self.assertEqual(obj.get('bar'), 'loaded!')
        self.assertEqual(obj.get('bar', 'not-loaded'), 'loaded!')
        self.assertRaises(AttributeError, obj.get, 'nothing')
        self.assertRaises(AttributeError, obj.get, 'nothing', 3)

    def test_object_inheritance(self):
        base_fields = []
        myobj_fields = ['foo', 'bar', 'missing', 'readonly', 'rel_object', 'rel_objects', 'mutable_default', 'timestamp'] + base_fields
        myobj3_fields = ['new_field']
        self.assertTrue(issubclass(TestSubclassedObject, MyObj))
        self.assertEqual(len(myobj_fields), len(MyObj.fields))
        self.assertEqual(set(myobj_fields), set(MyObj.fields.keys()))
        self.assertEqual(len(myobj_fields) + len(myobj3_fields), len(TestSubclassedObject.fields))
        self.assertEqual(set(myobj_fields) | set(myobj3_fields), set(TestSubclassedObject.fields.keys()))

    def test_obj_as_admin(self):
        self.skipTest('oslo.context does not support elevated()')
        obj = MyObj(context=self.context)

        def fake(*args, **kwargs):
            self.assertTrue(obj._context.is_admin)
        with mock.patch.object(obj, 'obj_reset_changes') as mock_fn:
            mock_fn.side_effect = fake
            with obj.obj_as_admin():
                obj.save()
            self.assertTrue(mock_fn.called)
        self.assertFalse(obj._context.is_admin)

    def test_get_changes(self):
        obj = MyObj()
        self.assertEqual({}, obj.obj_get_changes())
        obj.foo = 123
        self.assertEqual({'foo': 123}, obj.obj_get_changes())
        obj.bar = 'test'
        self.assertEqual({'foo': 123, 'bar': 'test'}, obj.obj_get_changes())
        obj.obj_reset_changes()
        self.assertEqual({}, obj.obj_get_changes())
        timestamp = datetime.datetime(2001, 1, 1, tzinfo=pytz.utc)
        with mock.patch.object(timeutils, 'utcnow') as mock_utcnow:
            mock_utcnow.return_value = timestamp
            obj.timestamp = timeutils.utcnow()
            self.assertEqual({'timestamp': timestamp}, obj.obj_get_changes())
        obj.obj_reset_changes()
        self.assertEqual({}, obj.obj_get_changes())
        timestamp = datetime.datetime(2001, 1, 1)
        with mock.patch.object(timeutils, 'utcnow') as mock_utcnow:
            mock_utcnow.return_value = timestamp
            obj.timestamp = timeutils.utcnow()
            self.assertRaises(TypeError, obj.obj_get_changes())
        obj.obj_reset_changes()
        self.assertEqual({}, obj.obj_get_changes())

    def test_obj_fields(self):

        class TestObj(base.VersionedObject):
            fields = {'foo': fields.Field(fields.Integer())}
            obj_extra_fields = ['bar']

            @property
            def bar(self):
                return 'this is bar'
        obj = TestObj()
        self.assertEqual(['foo', 'bar'], obj.obj_fields)

    def test_obj_context(self):

        class TestObj(base.VersionedObject):
            pass
        context = mock.Mock()
        obj = TestObj(context)
        self.assertEqual(context, obj.obj_context)
        new_context = mock.Mock()
        self.assertRaises(AttributeError, setattr, obj, 'obj_context', new_context)

    def test_obj_constructor(self):
        obj = MyObj(context=self.context, foo=123, bar='abc')
        self.assertEqual(123, obj.foo)
        self.assertEqual('abc', obj.bar)
        self.assertEqual(set(['foo', 'bar']), obj.obj_what_changed())

    def test_obj_read_only(self):
        obj = MyObj(context=self.context, foo=123, bar='abc')
        obj.readonly = 1
        self.assertRaises(exception.ReadOnlyFieldError, setattr, obj, 'readonly', 2)

    def test_obj_mutable_default(self):
        obj = MyObj(context=self.context, foo=123, bar='abc')
        obj.mutable_default = None
        obj.mutable_default.append('s1')
        self.assertEqual(obj.mutable_default, ['s1'])
        obj1 = MyObj(context=self.context, foo=123, bar='abc')
        obj1.mutable_default = None
        obj1.mutable_default.append('s2')
        self.assertEqual(obj1.mutable_default, ['s2'])

    def test_obj_mutable_default_set_default(self):
        obj1 = MyObj(context=self.context, foo=123, bar='abc')
        obj1.obj_set_defaults('mutable_default')
        self.assertEqual(obj1.mutable_default, [])
        obj1.mutable_default.append('s1')
        self.assertEqual(obj1.mutable_default, ['s1'])
        obj2 = MyObj(context=self.context, foo=123, bar='abc')
        obj2.obj_set_defaults('mutable_default')
        self.assertEqual(obj2.mutable_default, [])
        obj2.mutable_default.append('s2')
        self.assertEqual(obj2.mutable_default, ['s2'])

    def test_obj_repr(self):
        obj = MyObj(foo=123)
        self.assertEqual('MyObj(bar=<?>,foo=123,missing=<?>,mutable_default=<?>,readonly=<?>,rel_object=<?>,rel_objects=<?>,timestamp=<?>)', repr(obj))

    def test_obj_repr_sensitive(self):
        obj = MySensitiveObj(data="{'admin_password':'mypassword'}")
        self.assertEqual("MySensitiveObj(data='{'admin_password':'***'}')", repr(obj))
        obj2 = MySensitiveObj()
        self.assertEqual('MySensitiveObj(data=<?>)', repr(obj2))

    def test_obj_repr_unicode(self):
        obj = MyObj(bar='Ƒơơ')
        self.assertEqual("MyObj(bar='Ƒơơ',foo=<?>,missing=<?>,mutable_default=<?>,readonly=<?>,rel_object=<?>,rel_objects=<?>,timestamp=<?>)", repr(obj))

    def test_obj_make_obj_compatible_with_relationships(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(rel_object=subobj)
        obj.obj_relationships = {'rel_object': [('1.5', '1.1'), ('1.7', '1.2')]}
        primitive = obj.obj_to_primitive()['versioned_object.data']
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj._obj_make_obj_compatible(copy.copy(primitive), '1.8', 'rel_object')
            self.assertFalse(mock_compat.called)
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj._obj_make_obj_compatible(copy.copy(primitive), '1.7', 'rel_object')
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.2')
            self.assertEqual('1.2', primitive['rel_object']['versioned_object.version'])
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj._obj_make_obj_compatible(copy.copy(primitive), '1.6', 'rel_object')
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.1')
            self.assertEqual('1.1', primitive['rel_object']['versioned_object.version'])
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj._obj_make_obj_compatible(copy.copy(primitive), '1.5', 'rel_object')
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.1')
            self.assertEqual('1.1', primitive['rel_object']['versioned_object.version'])
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            _prim = copy.copy(primitive)
            obj._obj_make_obj_compatible(_prim, '1.4', 'rel_object')
            self.assertFalse(mock_compat.called)
            self.assertNotIn('rel_object', _prim)

    def test_obj_make_compatible_hits_sub_objects_with_rels(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(foo=123, rel_object=subobj)
        obj.obj_relationships = {'rel_object': [('1.0', '1.0')]}
        with mock.patch.object(obj, '_obj_make_obj_compatible') as mock_compat:
            obj.obj_make_compatible({'rel_object': 'foo'}, '1.10')
            mock_compat.assert_called_once_with({'rel_object': 'foo'}, '1.10', 'rel_object')

    def test_obj_make_compatible_skips_unset_sub_objects_with_rels(self):
        obj = MyObj(foo=123)
        obj.obj_relationships = {'rel_object': [('1.0', '1.0')]}
        with mock.patch.object(obj, '_obj_make_obj_compatible') as mock_compat:
            obj.obj_make_compatible({'rel_object': 'foo'}, '1.10')
            self.assertFalse(mock_compat.called)

    def test_obj_make_compatible_complains_about_missing_rel_rules(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(foo=123, rel_object=subobj)
        obj.obj_relationships = {}
        self.assertRaises(exception.ObjectActionError, obj.obj_make_compatible, {}, '1.0')

    def test_obj_make_compatible_handles_list_of_objects_with_rels(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(rel_objects=[subobj])
        obj.obj_relationships = {'rel_objects': [('1.0', '1.123')]}

        def fake_make_compat(primitive, version, **k):
            self.assertEqual('1.123', version)
            self.assertIn('baz', primitive)
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_mc:
            mock_mc.side_effect = fake_make_compat
            obj.obj_to_primitive('1.0')
            self.assertTrue(mock_mc.called)

    def test_obj_make_compatible_with_manifest(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(rel_object=subobj)
        obj.obj_relationships = {}
        orig_primitive = obj.obj_to_primitive()['versioned_object.data']
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            manifest = {'MyOwnedObject': '1.2'}
            primitive = copy.deepcopy(orig_primitive)
            obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.2')
            self.assertEqual('1.2', primitive['rel_object']['versioned_object.version'])
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            manifest = {'MyOwnedObject': '1.0'}
            primitive = copy.deepcopy(orig_primitive)
            obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.0')
            self.assertEqual('1.0', primitive['rel_object']['versioned_object.version'])
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            manifest = {}
            primitive = copy.deepcopy(orig_primitive)
            obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
            self.assertFalse(mock_compat.called)
            self.assertEqual('1.0', primitive['rel_object']['versioned_object.version'])

    def test_obj_make_compatible_with_manifest_subobj(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(rel_object=subobj)
        obj.obj_relationships = {}
        manifest = {'MyOwnedObject': '1.2'}
        primitive = obj.obj_to_primitive()['versioned_object.data']
        method = 'obj_make_compatible_from_manifest'
        with mock.patch.object(subobj, method) as mock_compat:
            obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
            mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.2', version_manifest=manifest)

    def test_obj_make_compatible_with_manifest_subobj_list(self):
        subobj = MyOwnedObject(baz=1)
        obj = MyObj(rel_objects=[subobj])
        obj.obj_relationships = {}
        manifest = {'MyOwnedObject': '1.2'}
        primitive = obj.obj_to_primitive()['versioned_object.data']
        method = 'obj_make_compatible_from_manifest'
        with mock.patch.object(subobj, method) as mock_compat:
            obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
            mock_compat.assert_called_once_with(primitive['rel_objects'][0]['versioned_object.data'], '1.2', version_manifest=manifest)

    def test_obj_make_compatible_removes_field_cleans_changes(self):

        @base.VersionedObjectRegistry.register_if(False)
        class TestObject(base.VersionedObject):
            VERSION = '1.1'
            fields = {'foo': fields.StringField(), 'bar': fields.StringField()}

            def obj_make_compatible(self, primitive, target_version):
                del primitive['bar']
        obj = TestObject(foo='test1', bar='test2')
        prim = obj.obj_to_primitive('1.0')
        self.assertEqual(['foo'], prim['versioned_object.changes'])

    def test_delattr(self):
        obj = MyObj(bar='foo')
        del obj.bar
        self.assertFalse(obj.obj_attr_is_set('bar'))
        self.assertEqual('loaded!', getattr(obj, 'bar'))

    def test_delattr_unset(self):
        obj = MyObj()
        self.assertRaises(AttributeError, delattr, obj, 'bar')

    def test_obj_make_compatible_on_list_base(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            VERSION = '1.1'
            fields = {'objects': fields.ListOfObjectsField('MyObj')}
        childobj = MyObj(foo=1)
        listobj = MyList(objects=[childobj])
        compat_func = 'obj_make_compatible_from_manifest'
        with mock.patch.object(childobj, compat_func) as mock_compat:
            listobj.obj_to_primitive(target_version='1.0')
            mock_compat.assert_called_once_with({'foo': 1}, '1.0', version_manifest=None)

    def test_comparable_objects(self):

        class NonVersionedObject(object):
            pass
        obj1 = MyComparableObj(foo=1)
        obj2 = MyComparableObj(foo=1)
        obj3 = MyComparableObj(foo=2)
        obj4 = NonVersionedObject()
        self.assertTrue(obj1 == obj2)
        self.assertFalse(obj1 == obj3)
        self.assertFalse(obj1 == obj4)
        self.assertNotEqual(obj1, None)

    def test_compound_clone(self):
        obj = MyCompoundObject()
        obj.foo = [1, 2, 3]
        obj.bar = {'a': 1, 'b': 2, 'c': 3}
        obj.baz = set([1, 2, 3])
        copy = obj.obj_clone()
        self.assertEqual(obj.foo, copy.foo)
        self.assertEqual(obj.bar, copy.bar)
        self.assertEqual(obj.baz, copy.baz)
        copy.foo.append('4')
        copy.bar.update(d='4')
        copy.baz.add('4')
        self.assertEqual([1, 2, 3, 4], copy.foo)
        self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'd': 4}, copy.bar)
        self.assertEqual(set([1, 2, 3, 4]), copy.baz)

    def test_obj_list_fields_modifications(self):

        @base.VersionedObjectRegistry.register
        class ObjWithList(base.VersionedObject):
            fields = {'list_field': fields.Field(fields.List(fields.Integer()))}
        obj = ObjWithList()

        def set_by_index(val):
            obj.list_field[0] = val

        def append(val):
            obj.list_field.append(val)

        def extend(val):
            obj.list_field.extend([val])

        def add(val):
            obj.list_field = obj.list_field + [val]

        def iadd(val):
            """Test += corner case

            a=a+b and a+=b use different magic methods under the hood:
            first one calls __add__ which clones initial value before the
            assignment, second one call __iadd__ which modifies the initial
            list.
            Assignment should cause coercing in both cases, but __iadd__ may
            corrupt the initial value even if the assignment fails.
            So it should be overridden as well, and this test is needed to
            verify it
            """
            obj.list_field += [val]

        def insert(val):
            obj.list_field.insert(0, val)

        def simple_slice(val):
            obj.list_field[:] = [val]

        def extended_slice(val):
            """Extended slice case

            Extended slice (and regular slices in py3) are handled differently
            thus needing a separate test
            """
            obj.list_field[::2] = [val]
        obj.list_field = ['42']
        set_by_index('1')
        append('2')
        extend('3')
        add('4')
        iadd('5')
        insert('0')
        self.assertEqual([0, 1, 2, 3, 4, 5], obj.list_field)
        simple_slice('10')
        self.assertEqual([10], obj.list_field)
        extended_slice('42')
        self.assertEqual([42], obj.list_field)
        obj.obj_reset_changes()
        self.assertRaises(ValueError, set_by_index, 'abc')
        self.assertRaises(ValueError, append, 'abc')
        self.assertRaises(ValueError, extend, 'abc')
        self.assertRaises(ValueError, add, 'abc')
        self.assertRaises(ValueError, iadd, 'abc')
        self.assertRaises(ValueError, insert, 'abc')
        self.assertRaises(ValueError, simple_slice, 'abc')
        self.assertRaises(ValueError, extended_slice, 'abc')
        self.assertEqual([42], obj.list_field)
        self.assertEqual({}, obj.obj_get_changes())

    def test_obj_dict_field_modifications(self):

        @base.VersionedObjectRegistry.register
        class ObjWithDict(base.VersionedObject):
            fields = {'dict_field': fields.Field(fields.Dict(fields.Integer()))}
        obj = ObjWithDict()
        obj.dict_field = {'1': 1, '3': 3, '4': 4}

        def set_by_key(key, value):
            obj.dict_field[key] = value

        def add_by_key(key, value):
            obj.dict_field[key] = value

        def update_w_dict(key, value):
            obj.dict_field.update({key: value})

        def update_w_kwargs(key, value):
            obj.dict_field.update(**{key: value})

        def setdefault(key, value):
            obj.dict_field.setdefault(key, value)
        set_by_key('1', '10')
        add_by_key('2', '20')
        update_w_dict('3', '30')
        update_w_kwargs('4', '40')
        setdefault('5', '50')
        self.assertEqual({'1': 10, '2': 20, '3': 30, '4': 40, '5': 50}, obj.dict_field)
        obj.obj_reset_changes()
        self.assertRaises(ValueError, set_by_key, 'key', 'abc')
        self.assertRaises(ValueError, add_by_key, 'other', 'abc')
        self.assertRaises(ValueError, update_w_dict, 'key', 'abc')
        self.assertRaises(ValueError, update_w_kwargs, 'key', 'abc')
        self.assertRaises(ValueError, setdefault, 'other', 'abc')
        self.assertEqual({'1': 10, '2': 20, '3': 30, '4': 40, '5': 50}, obj.dict_field)
        self.assertEqual({}, obj.obj_get_changes())

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