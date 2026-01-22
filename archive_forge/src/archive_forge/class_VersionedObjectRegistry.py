import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
class VersionedObjectRegistry(object):
    _registry = None

    def __new__(cls, *args, **kwargs):
        if not VersionedObjectRegistry._registry:
            VersionedObjectRegistry._registry = object.__new__(VersionedObjectRegistry, *args, **kwargs)
            VersionedObjectRegistry._registry._obj_classes = collections.defaultdict(list)
        self = object.__new__(cls, *args, **kwargs)
        self._obj_classes = VersionedObjectRegistry._registry._obj_classes
        return self

    def registration_hook(self, cls, index):
        pass

    def _register_class(self, cls):

        def _vers_tuple(obj):
            return vutils.convert_version_to_tuple(obj.VERSION)
        _make_class_properties(cls)
        obj_name = cls.obj_name()
        for i, obj in enumerate(self._obj_classes[obj_name]):
            self.registration_hook(cls, i)
            if cls.VERSION == obj.VERSION:
                self._obj_classes[obj_name][i] = cls
                break
            if _vers_tuple(cls) > _vers_tuple(obj):
                self._obj_classes[obj_name].insert(i, cls)
                break
        else:
            self._obj_classes[obj_name].append(cls)
            self.registration_hook(cls, 0)

    @classmethod
    def register(cls, obj_cls):
        registry = cls()
        registry._register_class(obj_cls)
        return obj_cls

    @classmethod
    def register_if(cls, condition):

        def wraps(obj_cls):
            if condition:
                obj_cls = cls.register(obj_cls)
            else:
                _make_class_properties(obj_cls)
            return obj_cls
        return wraps

    @classmethod
    def objectify(cls, obj_cls):
        return cls.register_if(False)(obj_cls)

    @classmethod
    def obj_classes(cls):
        registry = cls()
        return registry._obj_classes