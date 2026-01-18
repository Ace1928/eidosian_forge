import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___setuptools_pypa_io_en_latest_references_keywords_html__definitions_file_directive(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/file-directive', 'title': "'file:' directive", 'description': 'Value is read from a file (or list of files and then concatenated)', 'type': 'object', 'additionalProperties': False, 'properties': {'file': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]}}, 'required': ['file']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all((prop in data for prop in ['file'])):
            raise JsonSchemaValueException('' + (name_prefix or 'data') + " must contain ['file'] properties", value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/file-directive', 'title': "'file:' directive", 'description': 'Value is read from a file (or list of files and then concatenated)', 'type': 'object', 'additionalProperties': False, 'properties': {'file': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]}}, 'required': ['file']}, rule='required')
        data_keys = set(data.keys())
        if 'file' in data_keys:
            data_keys.remove('file')
            data__file = data['file']
            data__file_one_of_count7 = 0
            if data__file_one_of_count7 < 2:
                try:
                    if not isinstance(data__file, str):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.file must be string', value=data__file, name='' + (name_prefix or 'data') + '.file', definition={'type': 'string'}, rule='type')
                    data__file_one_of_count7 += 1
                except JsonSchemaValueException:
                    pass
            if data__file_one_of_count7 < 2:
                try:
                    if not isinstance(data__file, (list, tuple)):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.file must be array', value=data__file, name='' + (name_prefix or 'data') + '.file', definition={'type': 'array', 'items': {'type': 'string'}}, rule='type')
                    data__file_is_list = isinstance(data__file, (list, tuple))
                    if data__file_is_list:
                        data__file_len = len(data__file)
                        for data__file_x, data__file_item in enumerate(data__file):
                            if not isinstance(data__file_item, str):
                                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.file[{data__file_x}]'.format(**locals()) + ' must be string', value=data__file_item, name='' + (name_prefix or 'data') + '.file[{data__file_x}]'.format(**locals()) + '', definition={'type': 'string'}, rule='type')
                    data__file_one_of_count7 += 1
                except JsonSchemaValueException:
                    pass
            if data__file_one_of_count7 != 1:
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.file must be valid exactly by one definition' + (' (' + str(data__file_one_of_count7) + ' matches found)'), value=data__file, name='' + (name_prefix or 'data') + '.file', definition={'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]}, rule='oneOf')
        if data_keys:
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must not contain ' + str(data_keys) + ' properties', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/file-directive', 'title': "'file:' directive", 'description': 'Value is read from a file (or list of files and then concatenated)', 'type': 'object', 'additionalProperties': False, 'properties': {'file': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]}}, 'required': ['file']}, rule='additionalProperties')
    return data