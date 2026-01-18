import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___docs_python_org_3_install(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'$schema': 'http://json-schema.org/draft-07/schema', '$id': 'https://docs.python.org/3/install/', 'title': '``tool.distutils`` table', '$$description': ['Originally, ``distutils`` allowed developers to configure arguments for', '``setup.py`` scripts via `distutils configuration files', '<https://docs.python.org/3/install/#distutils-configuration-files>`_.', '``tool.distutils`` subtables could be used with the same purpose', '(NOT CURRENTLY IMPLEMENTED).'], 'type': 'object', 'properties': {'global': {'type': 'object', 'description': 'Global options applied to all ``distutils`` commands'}}, 'patternProperties': {'.+': {'type': 'object'}}, '$comment': 'TODO: Is there a practical way of making this schema more specific?'}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_keys = set(data.keys())
        if 'global' in data_keys:
            data_keys.remove('global')
            data__global = data['global']
            if not isinstance(data__global, dict):
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.global must be object', value=data__global, name='' + (name_prefix or 'data') + '.global', definition={'type': 'object', 'description': 'Global options applied to all ``distutils`` commands'}, rule='type')
        for data_key, data_val in data.items():
            if REGEX_PATTERNS['.+'].search(data_key):
                if data_key in data_keys:
                    data_keys.remove(data_key)
                if not isinstance(data_val, dict):
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + ' must be object', value=data_val, name='' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + '', definition={'type': 'object'}, rule='type')
    return data