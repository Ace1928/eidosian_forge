import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __GetTypeInfo(self, attrs, name_hint):
    """Return a TypeInfo object for attrs, creating one if needed."""
    type_ref = self.__names.ClassName(attrs.get('$ref'))
    type_name = attrs.get('type')
    if not (type_ref or type_name):
        raise ValueError('No type found for %s' % attrs)
    if type_ref:
        self.__AddIfUnknown(type_ref)
        return TypeInfo(type_name=type_ref, variant=messages.Variant.MESSAGE)
    if 'enum' in attrs:
        enum_name = '%sValuesEnum' % name_hint
        return self.__DeclareEnum(enum_name, attrs)
    if 'format' in attrs:
        type_info = self.PRIMITIVE_FORMAT_MAP.get(attrs['format'])
        if type_info is None:
            if type_name in self.PRIMITIVE_TYPE_INFO_MAP:
                return self.PRIMITIVE_TYPE_INFO_MAP[type_name]
            raise ValueError('Unknown type/format "%s"/"%s"' % (attrs['format'], type_name))
        if type_info.type_name.startswith(('apitools.base.protorpclite.message_types.', 'message_types.')):
            self.__AddImport('from %s import message_types as _message_types' % self.__protorpc_package)
        if type_info.type_name.startswith('extra_types.'):
            self.__AddImport('from %s import extra_types' % self.__base_files_package)
        return type_info
    if type_name in self.PRIMITIVE_TYPE_INFO_MAP:
        type_info = self.PRIMITIVE_TYPE_INFO_MAP[type_name]
        if type_info.type_name.startswith('extra_types.'):
            self.__AddImport('from %s import extra_types' % self.__base_files_package)
        return type_info
    if type_name == 'array':
        items = attrs.get('items')
        if not items:
            raise ValueError('Array type with no item type: %s' % attrs)
        entry_name_hint = self.__names.ClassName(items.get('title') or '%sListEntry' % name_hint)
        entry_label = self.__ComputeLabel(items)
        if entry_label == descriptor.FieldDescriptor.Label.REPEATED:
            parent_name = self.__names.ClassName(items.get('title') or name_hint)
            entry_type_name = self.__AddEntryType(entry_name_hint, items.get('items'), parent_name)
            return TypeInfo(type_name=entry_type_name, variant=messages.Variant.MESSAGE)
        return self.__GetTypeInfo(items, entry_name_hint)
    elif type_name == 'any':
        self.__AddImport('from %s import extra_types' % self.__base_files_package)
        return self.PRIMITIVE_TYPE_INFO_MAP['any']
    elif type_name == 'object':
        if not name_hint:
            raise ValueError('Cannot create subtype without some name hint')
        schema = dict(attrs)
        schema['id'] = name_hint
        self.AddDescriptorFromSchema(name_hint, schema)
        self.__AddIfUnknown(name_hint)
        return TypeInfo(type_name=name_hint, variant=messages.Variant.MESSAGE)
    raise ValueError('Unknown type: %s' % type_name)