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
class VersionedObject(object):
    """Base class and object factory.

    This forms the base of all objects that can be remoted or instantiated
    via RPC. Simply defining a class that inherits from this base class
    will make it remotely instantiatable. Objects should implement the
    necessary "get" classmethod routines as well as "save" object methods
    as appropriate.
    """
    indirection_api = None
    VERSION = '1.0'
    OBJ_SERIAL_NAMESPACE = 'versioned_object'
    OBJ_PROJECT_NAMESPACE = 'versionedobjects'
    fields = {}
    obj_extra_fields = []
    obj_relationships = {}

    def __init__(self, context=None, **kwargs):
        self._changed_fields = set()
        self._context = context
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __repr__(self):
        repr_str = '%s(%s)' % (self.obj_name(), ','.join(['%s=%s' % (name, self.obj_attr_is_set(name) and field.stringify(getattr(self, name)) or '<?>') for name, field in sorted(self.fields.items())]))
        return repr_str

    def __contains__(self, name):
        try:
            return self.obj_attr_is_set(name)
        except AttributeError:
            return False

    @classmethod
    def to_json_schema(cls):
        obj_name = cls.obj_name()
        schema = {'$schema': 'http://json-schema.org/draft-04/schema#', 'title': obj_name}
        schema.update(obj_fields.Object(obj_name).get_schema())
        return schema

    @classmethod
    def obj_name(cls):
        """Return the object's name

        Return a canonical name for this object which will be used over
        the wire for remote hydration.
        """
        return cls.__name__

    @classmethod
    def _obj_primitive_key(cls, field):
        return '%s.%s' % (cls.OBJ_SERIAL_NAMESPACE, field)

    @classmethod
    def _obj_primitive_field(cls, primitive, field, default=obj_fields.UnspecifiedDefault):
        key = cls._obj_primitive_key(field)
        if default == obj_fields.UnspecifiedDefault:
            return primitive[key]
        else:
            return primitive.get(key, default)

    @classmethod
    def obj_class_from_name(cls, objname, objver):
        """Returns a class from the registry based on a name and version."""
        if objname not in VersionedObjectRegistry.obj_classes():
            (LOG.error('Unable to instantiate unregistered object type %(objtype)s'), dict(objtype=objname))
            raise exception.UnsupportedObjectError(objtype=objname)
        compatible_match = None
        for objclass in VersionedObjectRegistry.obj_classes()[objname]:
            if objclass.VERSION == objver:
                return objclass
            if not compatible_match and vutils.is_compatible(objver, objclass.VERSION):
                compatible_match = objclass
        if compatible_match:
            return compatible_match
        latest_ver = VersionedObjectRegistry.obj_classes()[objname][0].VERSION
        raise exception.IncompatibleObjectVersion(objname=objname, objver=objver, supported=latest_ver)

    @classmethod
    def _obj_from_primitive(cls, context, objver, primitive):
        self = cls()
        self._context = context
        self.VERSION = objver
        objdata = cls._obj_primitive_field(primitive, 'data')
        changes = cls._obj_primitive_field(primitive, 'changes', [])
        for name, field in self.fields.items():
            if name in objdata:
                setattr(self, name, field.from_primitive(self, name, objdata[name]))
        self._changed_fields = set([x for x in changes if x in self.fields])
        return self

    @classmethod
    def obj_from_primitive(cls, primitive, context=None):
        """Object field-by-field hydration."""
        objns = cls._obj_primitive_field(primitive, 'namespace')
        objname = cls._obj_primitive_field(primitive, 'name')
        objver = cls._obj_primitive_field(primitive, 'version')
        if objns != cls.OBJ_PROJECT_NAMESPACE:
            raise exception.UnsupportedObjectError(objtype='%s.%s' % (objns, objname))
        objclass = cls.obj_class_from_name(objname, objver)
        return objclass._obj_from_primitive(context, objver, primitive)

    def __deepcopy__(self, memo):
        """Efficiently make a deep copy of this object."""
        nobj = self.__class__()
        memo[id(self)] = nobj
        nobj._context = self._context
        for name in self.fields:
            if self.obj_attr_is_set(name):
                nval = copy.deepcopy(getattr(self, name), memo)
                setattr(nobj, name, nval)
        nobj._changed_fields = set(self._changed_fields)
        return nobj

    def obj_clone(self):
        """Create a copy."""
        return copy.deepcopy(self)

    def _obj_relationship_for(self, field, target_version):
        if not hasattr(self, '_obj_version_manifest') or self._obj_version_manifest is None:
            try:
                return self.obj_relationships[field]
            except KeyError:
                raise exception.ObjectActionError(action='obj_make_compatible', reason='No rule for %s' % field)
        objname = self.fields[field].objname
        if objname not in self._obj_version_manifest:
            return
        return [(target_version, self._obj_version_manifest[objname])]

    def _obj_make_obj_compatible(self, primitive, target_version, field):
        """Backlevel a sub-object based on our versioning rules.

        This is responsible for backporting objects contained within
        this object's primitive according to a set of rules we
        maintain about version dependencies between objects. This
        requires that the obj_relationships table in this object is
        correct and up-to-date.

        :param:primitive: The primitive version of this object
        :param:target_version: The version string requested for this object
        :param:field: The name of the field in this object containing the
                      sub-object to be backported
        """
        relationship_map = self._obj_relationship_for(field, target_version)
        if not relationship_map:
            return
        try:
            _get_subobject_version(target_version, relationship_map, lambda ver: _do_subobject_backport(ver, self, field, primitive))
        except exception.TargetBeforeSubobjectExistedException:
            del primitive[field]

    def obj_make_compatible(self, primitive, target_version):
        """Make an object representation compatible with a target version.

        This is responsible for taking the primitive representation of
        an object and making it suitable for the given target_version.
        This may mean converting the format of object attributes, removing
        attributes that have been added since the target version, etc. In
        general:

        - If a new version of an object adds a field, this routine
          should remove it for older versions.
        - If a new version changed or restricted the format of a field, this
          should convert it back to something a client knowing only of the
          older version will tolerate.
        - If an object that this object depends on is bumped, then this
          object should also take a version bump. Then, this routine should
          backlevel the dependent object (by calling its obj_make_compatible())
          if the requested version of this object is older than the version
          where the new dependent object was added.

        :param primitive: The result of :meth:`obj_to_primitive`
        :param target_version: The version string requested by the recipient
                               of the object
        :raises: :exc:`oslo_versionedobjects.exception.UnsupportedObjectError`
                 if conversion is not possible for some reason
        """
        for key, field in self.fields.items():
            if not isinstance(field, (obj_fields.ObjectField, obj_fields.ListOfObjectsField)):
                continue
            if not self.obj_attr_is_set(key):
                continue
            self._obj_make_obj_compatible(primitive, target_version, key)

    def obj_make_compatible_from_manifest(self, primitive, target_version, version_manifest):
        self._obj_version_manifest = version_manifest
        try:
            return self.obj_make_compatible(primitive, target_version)
        finally:
            delattr(self, '_obj_version_manifest')

    def obj_to_primitive(self, target_version=None, version_manifest=None):
        """Simple base-case dehydration.

        This calls to_primitive() for each item in fields.
        """
        if target_version is None:
            target_version = self.VERSION
        if vutils.convert_version_to_tuple(target_version) > vutils.convert_version_to_tuple(self.VERSION):
            raise exception.InvalidTargetVersion(version=target_version)
        primitive = dict()
        for name, field in self.fields.items():
            if self.obj_attr_is_set(name):
                primitive[name] = field.to_primitive(self, name, getattr(self, name))
        if target_version != self.VERSION or version_manifest:
            self.obj_make_compatible_from_manifest(primitive, target_version, version_manifest)
        obj = {self._obj_primitive_key('name'): self.obj_name(), self._obj_primitive_key('namespace'): self.OBJ_PROJECT_NAMESPACE, self._obj_primitive_key('version'): target_version, self._obj_primitive_key('data'): primitive}
        if self.obj_what_changed():
            what_changed = self.obj_what_changed()
            changes = [field for field in what_changed if field in primitive]
            if changes:
                obj[self._obj_primitive_key('changes')] = changes
        return obj

    def obj_set_defaults(self, *attrs):
        if not attrs:
            attrs = [name for name, field in self.fields.items() if field.default != obj_fields.UnspecifiedDefault]
        for attr in attrs:
            default = copy.deepcopy(self.fields[attr].default)
            if default is obj_fields.UnspecifiedDefault:
                raise exception.ObjectActionError(action='set_defaults', reason='No default set for field %s' % attr)
            if not self.obj_attr_is_set(attr):
                setattr(self, attr, default)

    def obj_load_attr(self, attrname):
        """Load an additional attribute from the real object.

        This should load self.$attrname and cache any data that might
        be useful for future load operations.
        """
        raise NotImplementedError(_("Cannot load '%s' in the base class") % attrname)

    def save(self, context):
        """Save the changed fields back to the store.

        This is optional for subclasses, but is presented here in the base
        class for consistency among those that do.
        """
        raise NotImplementedError(_('Cannot save anything in the base class'))

    def obj_what_changed(self):
        """Returns a set of fields that have been modified."""
        changes = set([field for field in self._changed_fields if field in self.fields])
        for field in self.fields:
            if self.obj_attr_is_set(field) and isinstance(getattr(self, field), VersionedObject) and getattr(self, field).obj_what_changed():
                changes.add(field)
        return changes

    def obj_get_changes(self):
        """Returns a dict of changed fields and their new values."""
        changes = {}
        for key in self.obj_what_changed():
            changes[key] = getattr(self, key)
        return changes

    def obj_reset_changes(self, fields=None, recursive=False):
        """Reset the list of fields that have been changed.

        :param fields: List of fields to reset, or "all" if None.
        :param recursive: Call obj_reset_changes(recursive=True) on
                          any sub-objects within the list of fields
                          being reset.

        This is NOT "revert to previous values".

        Specifying fields on recursive resets will only be honored at the top
        level. Everything below the top will reset all.
        """
        if recursive:
            for field in self.obj_get_changes():
                if fields and field not in fields:
                    continue
                if not self.obj_attr_is_set(field):
                    continue
                value = getattr(self, field)
                if value is None:
                    continue
                if isinstance(self.fields[field], obj_fields.ObjectField):
                    value.obj_reset_changes(recursive=True)
                elif isinstance(self.fields[field], obj_fields.ListOfObjectsField):
                    for thing in value:
                        thing.obj_reset_changes(recursive=True)
        if fields:
            self._changed_fields -= set(fields)
        else:
            self._changed_fields.clear()

    def obj_attr_is_set(self, attrname):
        """Test object to see if attrname is present.

        Returns True if the named attribute has a value set, or
        False if not. Raises AttributeError if attrname is not
        a valid attribute for this object.
        """
        if attrname not in self.obj_fields:
            raise AttributeError(_("%(objname)s object has no attribute '%(attrname)s'") % {'objname': self.obj_name(), 'attrname': attrname})
        return hasattr(self, _get_attrname(attrname))

    @property
    def obj_fields(self):
        return list(self.fields.keys()) + self.obj_extra_fields

    @property
    def obj_context(self):
        return self._context