import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class BaseXMLResponseParser(ResponseParser):

    def __init__(self, timestamp_parser=None, blob_parser=None):
        super().__init__(timestamp_parser, blob_parser)
        self._namespace_re = re.compile('{.*}')

    def _handle_map(self, shape, node):
        parsed = {}
        key_shape = shape.key
        value_shape = shape.value
        key_location_name = key_shape.serialization.get('name') or 'key'
        value_location_name = value_shape.serialization.get('name') or 'value'
        if shape.serialization.get('flattened') and (not isinstance(node, list)):
            node = [node]
        for keyval_node in node:
            for single_pair in keyval_node:
                tag_name = self._node_tag(single_pair)
                if tag_name == key_location_name:
                    key_name = self._parse_shape(key_shape, single_pair)
                elif tag_name == value_location_name:
                    val_name = self._parse_shape(value_shape, single_pair)
                else:
                    raise ResponseParserError('Unknown tag: %s' % tag_name)
            parsed[key_name] = val_name
        return parsed

    def _node_tag(self, node):
        return self._namespace_re.sub('', node.tag)

    def _handle_list(self, shape, node):
        if shape.serialization.get('flattened') and (not isinstance(node, list)):
            node = [node]
        return super()._handle_list(shape, node)

    def _handle_structure(self, shape, node):
        parsed = {}
        members = shape.members
        if shape.metadata.get('exception', False):
            node = self._get_error_root(node)
        xml_dict = self._build_name_to_xml_node(node)
        if self._has_unknown_tagged_union_member(shape, xml_dict):
            tag = self._get_first_key(xml_dict)
            return self._handle_unknown_tagged_union_member(tag)
        for member_name in members:
            member_shape = members[member_name]
            if 'location' in member_shape.serialization or member_shape.serialization.get('eventheader'):
                continue
            xml_name = self._member_key_name(member_shape, member_name)
            member_node = xml_dict.get(xml_name)
            if member_node is not None:
                parsed[member_name] = self._parse_shape(member_shape, member_node)
            elif member_shape.serialization.get('xmlAttribute'):
                attribs = {}
                location_name = member_shape.serialization['name']
                for key, value in node.attrib.items():
                    new_key = self._namespace_re.sub(location_name.split(':')[0] + ':', key)
                    attribs[new_key] = value
                if location_name in attribs:
                    parsed[member_name] = attribs[location_name]
        return parsed

    def _get_error_root(self, original_root):
        if self._node_tag(original_root) == 'ErrorResponse':
            for child in original_root:
                if self._node_tag(child) == 'Error':
                    return child
        return original_root

    def _member_key_name(self, shape, member_name):
        if shape.type_name == 'list' and shape.serialization.get('flattened'):
            list_member_serialized_name = shape.member.serialization.get('name')
            if list_member_serialized_name is not None:
                return list_member_serialized_name
        serialized_name = shape.serialization.get('name')
        if serialized_name is not None:
            return serialized_name
        return member_name

    def _build_name_to_xml_node(self, parent_node):
        if isinstance(parent_node, list):
            return self._build_name_to_xml_node(parent_node[0])
        xml_dict = {}
        for item in parent_node:
            key = self._node_tag(item)
            if key in xml_dict:
                if isinstance(xml_dict[key], list):
                    xml_dict[key].append(item)
                else:
                    xml_dict[key] = [xml_dict[key], item]
            else:
                xml_dict[key] = item
        return xml_dict

    def _parse_xml_string_to_dom(self, xml_string):
        try:
            parser = ETree.XMLParser(target=ETree.TreeBuilder(), encoding=self.DEFAULT_ENCODING)
            parser.feed(xml_string)
            root = parser.close()
        except XMLParseError as e:
            raise ResponseParserError('Unable to parse response (%s), invalid XML received. Further retries may succeed:\n%s' % (e, xml_string))
        return root

    def _replace_nodes(self, parsed):
        for key, value in parsed.items():
            if list(value):
                sub_dict = self._build_name_to_xml_node(value)
                parsed[key] = self._replace_nodes(sub_dict)
            else:
                parsed[key] = value.text
        return parsed

    @_text_content
    def _handle_boolean(self, shape, text):
        if text == 'true':
            return True
        else:
            return False

    @_text_content
    def _handle_float(self, shape, text):
        return float(text)

    @_text_content
    def _handle_timestamp(self, shape, text):
        return self._timestamp_parser(text)

    @_text_content
    def _handle_integer(self, shape, text):
        return int(text)

    @_text_content
    def _handle_string(self, shape, text):
        return text

    @_text_content
    def _handle_blob(self, shape, text):
        return self._blob_parser(text)
    _handle_character = _handle_string
    _handle_double = _handle_float
    _handle_long = _handle_integer