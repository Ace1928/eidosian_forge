import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
class RestXMLSerializer(BaseRestSerializer):
    TIMESTAMP_FORMAT = 'iso8601'

    def _serialize_body_params(self, params, shape):
        root_name = shape.serialization['name']
        pseudo_root = ElementTree.Element('')
        self._serialize(shape, params, pseudo_root, root_name)
        real_root = list(pseudo_root)[0]
        return ElementTree.tostring(real_root, encoding=self.DEFAULT_ENCODING)

    def _serialize(self, shape, params, xmlnode, name):
        method = getattr(self, '_serialize_type_%s' % shape.type_name, self._default_serialize)
        method(xmlnode, params, shape, name)

    def _serialize_type_structure(self, xmlnode, params, shape, name):
        structure_node = ElementTree.SubElement(xmlnode, name)
        if 'xmlNamespace' in shape.serialization:
            namespace_metadata = shape.serialization['xmlNamespace']
            attribute_name = 'xmlns'
            if namespace_metadata.get('prefix'):
                attribute_name += ':%s' % namespace_metadata['prefix']
            structure_node.attrib[attribute_name] = namespace_metadata['uri']
        for key, value in params.items():
            member_shape = shape.members[key]
            member_name = member_shape.serialization.get('name', key)
            if value is None:
                return
            if member_shape.serialization.get('xmlAttribute'):
                xml_attribute_name = member_shape.serialization['name']
                structure_node.attrib[xml_attribute_name] = value
                continue
            self._serialize(member_shape, value, structure_node, member_name)

    def _serialize_type_list(self, xmlnode, params, shape, name):
        member_shape = shape.member
        if shape.serialization.get('flattened'):
            element_name = name
            list_node = xmlnode
        else:
            element_name = member_shape.serialization.get('name', 'member')
            list_node = ElementTree.SubElement(xmlnode, name)
        for item in params:
            self._serialize(member_shape, item, list_node, element_name)

    def _serialize_type_map(self, xmlnode, params, shape, name):
        node = ElementTree.SubElement(xmlnode, name)
        for key, value in params.items():
            entry_node = ElementTree.SubElement(node, 'entry')
            key_name = self._get_serialized_name(shape.key, default_name='key')
            val_name = self._get_serialized_name(shape.value, default_name='value')
            self._serialize(shape.key, key, entry_node, key_name)
            self._serialize(shape.value, value, entry_node, val_name)

    def _serialize_type_boolean(self, xmlnode, params, shape, name):
        node = ElementTree.SubElement(xmlnode, name)
        if params:
            str_value = 'true'
        else:
            str_value = 'false'
        node.text = str_value

    def _serialize_type_blob(self, xmlnode, params, shape, name):
        node = ElementTree.SubElement(xmlnode, name)
        node.text = self._get_base64(params)

    def _serialize_type_timestamp(self, xmlnode, params, shape, name):
        node = ElementTree.SubElement(xmlnode, name)
        node.text = self._convert_timestamp_to_str(params, shape.serialization.get('timestampFormat'))

    def _default_serialize(self, xmlnode, params, shape, name):
        node = ElementTree.SubElement(xmlnode, name)
        node.text = str(params)