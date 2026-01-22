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
def _make_class_properties(cls):
    cls.fields = dict(cls.fields)
    for supercls in cls.mro()[1:-1]:
        if not hasattr(supercls, 'fields'):
            continue
        for name, field in supercls.fields.items():
            if name not in cls.fields:
                cls.fields[name] = field
    for name, field in cls.fields.items():
        if not isinstance(field, obj_fields.Field):
            raise exception.ObjectFieldInvalid(field=name, objname=cls.obj_name())

        def getter(self, name=name):
            attrname = _get_attrname(name)
            if not hasattr(self, attrname):
                self.obj_load_attr(name)
            return getattr(self, attrname)

        def setter(self, value, name=name, field=field):
            attrname = _get_attrname(name)
            field_value = field.coerce(self, name, value)
            if field.read_only and hasattr(self, attrname):
                if getattr(self, attrname) != field_value:
                    raise exception.ReadOnlyFieldError(field=name)
                else:
                    return
            self._changed_fields.add(name)
            try:
                return setattr(self, attrname, field_value)
            except Exception:
                with excutils.save_and_reraise_exception():
                    attr = '%s.%s' % (self.obj_name(), name)
                    LOG.exception('Error setting %(attr)s', {'attr': attr})

        def deleter(self, name=name):
            attrname = _get_attrname(name)
            if not hasattr(self, attrname):
                raise AttributeError("No such attribute `%s'" % name)
            delattr(self, attrname)
        setattr(cls, name, property(getter, setter, deleter))