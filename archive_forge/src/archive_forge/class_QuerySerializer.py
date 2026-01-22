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
class QuerySerializer(Serializer):
    TIMESTAMP_FORMAT = 'iso8601'

    def serialize_to_request(self, parameters, operation_model):
        shape = operation_model.input_shape
        serialized = self._create_default_request()
        serialized['method'] = operation_model.http.get('method', self.DEFAULT_METHOD)
        serialized['headers'] = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        body_params = self.MAP_TYPE()
        body_params['Action'] = operation_model.name
        body_params['Version'] = operation_model.metadata['apiVersion']
        if shape is not None:
            self._serialize(body_params, parameters, shape)
        serialized['body'] = body_params
        host_prefix = self._expand_host_prefix(parameters, operation_model)
        if host_prefix is not None:
            serialized['host_prefix'] = host_prefix
        return serialized

    def _serialize(self, serialized, value, shape, prefix=''):
        method = getattr(self, f'_serialize_type_{shape.type_name}', self._default_serialize)
        method(serialized, value, shape, prefix=prefix)

    def _serialize_type_structure(self, serialized, value, shape, prefix=''):
        members = shape.members
        for key, value in value.items():
            member_shape = members[key]
            member_prefix = self._get_serialized_name(member_shape, key)
            if prefix:
                member_prefix = f'{prefix}.{member_prefix}'
            self._serialize(serialized, value, member_shape, member_prefix)

    def _serialize_type_list(self, serialized, value, shape, prefix=''):
        if not value:
            serialized[prefix] = ''
            return
        if self._is_shape_flattened(shape):
            list_prefix = prefix
            if shape.member.serialization.get('name'):
                name = self._get_serialized_name(shape.member, default_name='')
                list_prefix = '.'.join(prefix.split('.')[:-1] + [name])
        else:
            list_name = shape.member.serialization.get('name', 'member')
            list_prefix = f'{prefix}.{list_name}'
        for i, element in enumerate(value, 1):
            element_prefix = f'{list_prefix}.{i}'
            element_shape = shape.member
            self._serialize(serialized, element, element_shape, element_prefix)

    def _serialize_type_map(self, serialized, value, shape, prefix=''):
        if self._is_shape_flattened(shape):
            full_prefix = prefix
        else:
            full_prefix = '%s.entry' % prefix
        template = full_prefix + '.{i}.{suffix}'
        key_shape = shape.key
        value_shape = shape.value
        key_suffix = self._get_serialized_name(key_shape, default_name='key')
        value_suffix = self._get_serialized_name(value_shape, 'value')
        for i, key in enumerate(value, 1):
            key_prefix = template.format(i=i, suffix=key_suffix)
            value_prefix = template.format(i=i, suffix=value_suffix)
            self._serialize(serialized, key, key_shape, key_prefix)
            self._serialize(serialized, value[key], value_shape, value_prefix)

    def _serialize_type_blob(self, serialized, value, shape, prefix=''):
        serialized[prefix] = self._get_base64(value)

    def _serialize_type_timestamp(self, serialized, value, shape, prefix=''):
        serialized[prefix] = self._convert_timestamp_to_str(value, shape.serialization.get('timestampFormat'))

    def _serialize_type_boolean(self, serialized, value, shape, prefix=''):
        if value:
            serialized[prefix] = 'true'
        else:
            serialized[prefix] = 'false'

    def _default_serialize(self, serialized, value, shape, prefix=''):
        serialized[prefix] = value

    def _is_shape_flattened(self, shape):
        return shape.serialization.get('flattened')