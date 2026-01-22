from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
class ObjectVersionChecker(object):

    def __init__(self, obj_classes=base.VersionedObjectRegistry.obj_classes()):
        self.obj_classes = obj_classes

    def _find_remotable_method(self, cls, thing, parent_was_remotable=False):
        """Follow a chain of remotable things down to the original function."""
        if isinstance(thing, classmethod):
            return self._find_remotable_method(cls, thing.__get__(None, cls))
        elif (inspect.ismethod(thing) or inspect.isfunction(thing)) and hasattr(thing, 'remotable'):
            return self._find_remotable_method(cls, thing.original_fn, parent_was_remotable=True)
        elif parent_was_remotable:
            return thing
        else:
            return None

    def _get_fingerprint(self, obj_name, extra_data_func=None):
        obj_class = self.obj_classes[obj_name][0]
        obj_fields = list(obj_class.fields.items())
        obj_fields.sort()
        methods = []
        for name in dir(obj_class):
            thing = getattr(obj_class, name)
            if inspect.ismethod(thing) or inspect.isfunction(thing) or isinstance(thing, classmethod):
                method = self._find_remotable_method(obj_class, thing)
                if method:
                    methods.append((name, get_method_spec(method)))
        methods.sort()
        if hasattr(obj_class, 'child_versions'):
            relevant_data = (obj_fields, methods, OrderedDict(sorted(obj_class.child_versions.items())))
        else:
            relevant_data = (obj_fields, methods)
        if extra_data_func:
            relevant_data += extra_data_func(obj_class)
        fingerprint = '%s-%s' % (obj_class.VERSION, md5(bytes(repr(relevant_data).encode()), usedforsecurity=False).hexdigest())
        return fingerprint

    def get_hashes(self, extra_data_func=None):
        """Return a dict of computed object hashes.

        :param extra_data_func: a function that is given the object class
                                which gathers more relevant data about the
                                class that is needed in versioning. Returns
                                a tuple containing the extra data bits.
        """
        fingerprints = {}
        for obj_name in sorted(self.obj_classes):
            fingerprints[obj_name] = self._get_fingerprint(obj_name, extra_data_func=extra_data_func)
        return fingerprints

    def test_hashes(self, expected_hashes, extra_data_func=None):
        fingerprints = self.get_hashes(extra_data_func=extra_data_func)
        stored = set(expected_hashes.items())
        computed = set(fingerprints.items())
        changed = stored.symmetric_difference(computed)
        expected = {}
        actual = {}
        for name, hash in changed:
            expected[name] = expected_hashes.get(name)
            actual[name] = fingerprints.get(name)
        return (expected, actual)

    def _get_dependencies(self, tree, obj_class):
        obj_name = obj_class.obj_name()
        if obj_name in tree:
            return
        for name, field in obj_class.fields.items():
            if isinstance(field._type, fields.Object):
                sub_obj_name = field._type._obj_name
                sub_obj_class = self.obj_classes[sub_obj_name][0]
                self._get_dependencies(tree, sub_obj_class)
                tree.setdefault(obj_name, {})
                tree[obj_name][sub_obj_name] = sub_obj_class.VERSION

    def get_dependency_tree(self):
        tree = {}
        for obj_name in self.obj_classes.keys():
            self._get_dependencies(tree, self.obj_classes[obj_name][0])
        return tree

    def test_relationships(self, expected_tree):
        actual_tree = self.get_dependency_tree()
        stored = set([(x, str(y)) for x, y in expected_tree.items()])
        computed = set([(x, str(y)) for x, y in actual_tree.items()])
        changed = stored.symmetric_difference(computed)
        expected = {}
        actual = {}
        for name, deps in changed:
            expected[name] = expected_tree.get(name)
            actual[name] = actual_tree.get(name)
        return (expected, actual)

    def _test_object_compatibility(self, obj_class, manifest=None, init_args=None, init_kwargs=None):
        init_args = init_args or []
        init_kwargs = init_kwargs or {}
        version = vutils.convert_version_to_tuple(obj_class.VERSION)
        kwargs = {'version_manifest': manifest} if manifest else {}
        for n in range(version[1] + 1):
            test_version = '%d.%d' % (version[0], n)
            LOG.debug('testing obj: %s version: %s' % (obj_class.obj_name(), test_version))
            kwargs['target_version'] = test_version
            obj_class(*init_args, **init_kwargs).obj_to_primitive(**kwargs)

    def test_compatibility_routines(self, use_manifest=False, init_args=None, init_kwargs=None):
        """Test obj_make_compatible() on all object classes.

        :param use_manifest: a boolean that determines if the version
                             manifest should be passed to obj_make_compatible
        :param init_args: a dictionary of the format {obj_class: [arg1, arg2]}
                          that will be used to pass arguments to init on the
                          given obj_class. If no args are needed, the
                          obj_class does not need to be added to the dict
        :param init_kwargs: a dictionary of the format
                            {obj_class: {'kwarg1': val1}} that will be used to
                            pass kwargs to init on the given obj_class. If no
                            kwargs are needed, the obj_class does not need to
                            be added to the dict
        """
        init_args = init_args or {}
        init_kwargs = init_kwargs or {}
        for obj_name in self.obj_classes:
            obj_classes = self.obj_classes[obj_name]
            if use_manifest:
                manifest = base.obj_tree_get_versions(obj_name)
            else:
                manifest = None
            for obj_class in obj_classes:
                args_for_init = init_args.get(obj_class, [])
                kwargs_for_init = init_kwargs.get(obj_class, {})
                self._test_object_compatibility(obj_class, manifest=manifest, init_args=args_for_init, init_kwargs=kwargs_for_init)

    def _test_relationships_in_order(self, obj_class):
        for field, versions in obj_class.obj_relationships.items():
            last_my_version = (0, 0)
            last_child_version = (0, 0)
            for my_version, child_version in versions:
                _my_version = vutils.convert_version_to_tuple(my_version)
                _ch_version = vutils.convert_version_to_tuple(child_version)
                if not (last_my_version < _my_version and last_child_version <= _ch_version):
                    raise AssertionError('Object %s relationship %s->%s for field %s is out of order' % (obj_class.obj_name(), my_version, child_version, field))
                last_my_version = _my_version
                last_child_version = _ch_version

    def test_relationships_in_order(self):
        for obj_name in self.obj_classes:
            obj_classes = self.obj_classes[obj_name]
            for obj_class in obj_classes:
                self._test_relationships_in_order(obj_class)