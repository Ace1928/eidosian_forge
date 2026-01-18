import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___packaging_python_org_en_latest_specifications_declaring_project_metadata___definitions_entry_point_group(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/entry-point-group', 'title': 'Entry-points', 'type': 'object', '$$description': ['Entry-points are grouped together to indicate what sort of capabilities they', 'provide.', 'See the `packaging guides', '<https://packaging.python.org/specifications/entry-points/>`_', 'and `setuptools docs', '<https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_', 'for more information.'], 'propertyNames': {'format': 'python-entrypoint-name'}, 'additionalProperties': False, 'patternProperties': {'^.+$': {'type': 'string', '$$description': ['Reference to a Python object. It is either in the form', '``importable.module``, or ``importable.module:object.attr``.'], 'format': 'python-entrypoint-reference', '$comment': 'https://packaging.python.org/specifications/entry-points/'}}}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_keys = set(data.keys())
        for data_key, data_val in data.items():
            if REGEX_PATTERNS['^.+$'].search(data_key):
                if data_key in data_keys:
                    data_keys.remove(data_key)
                if not isinstance(data_val, str):
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + ' must be string', value=data_val, name='' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + '', definition={'type': 'string', '$$description': ['Reference to a Python object. It is either in the form', '``importable.module``, or ``importable.module:object.attr``.'], 'format': 'python-entrypoint-reference', '$comment': 'https://packaging.python.org/specifications/entry-points/'}, rule='type')
                if isinstance(data_val, str):
                    if not custom_formats['python-entrypoint-reference'](data_val):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + ' must be python-entrypoint-reference', value=data_val, name='' + (name_prefix or 'data') + '.{data_key}'.format(**locals()) + '', definition={'type': 'string', '$$description': ['Reference to a Python object. It is either in the form', '``importable.module``, or ``importable.module:object.attr``.'], 'format': 'python-entrypoint-reference', '$comment': 'https://packaging.python.org/specifications/entry-points/'}, rule='format')
        if data_keys:
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must not contain ' + str(data_keys) + ' properties', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/entry-point-group', 'title': 'Entry-points', 'type': 'object', '$$description': ['Entry-points are grouped together to indicate what sort of capabilities they', 'provide.', 'See the `packaging guides', '<https://packaging.python.org/specifications/entry-points/>`_', 'and `setuptools docs', '<https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_', 'for more information.'], 'propertyNames': {'format': 'python-entrypoint-name'}, 'additionalProperties': False, 'patternProperties': {'^.+$': {'type': 'string', '$$description': ['Reference to a Python object. It is either in the form', '``importable.module``, or ``importable.module:object.attr``.'], 'format': 'python-entrypoint-reference', '$comment': 'https://packaging.python.org/specifications/entry-points/'}}}, rule='additionalProperties')
        data_len = len(data)
        if data_len != 0:
            data_property_names = True
            for data_key in data:
                try:
                    if isinstance(data_key, str):
                        if not custom_formats['python-entrypoint-name'](data_key):
                            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be python-entrypoint-name', value=data_key, name='' + (name_prefix or 'data') + '', definition={'format': 'python-entrypoint-name'}, rule='format')
                except JsonSchemaValueException:
                    data_property_names = False
            if not data_property_names:
                raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be named by propertyName definition', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/entry-point-group', 'title': 'Entry-points', 'type': 'object', '$$description': ['Entry-points are grouped together to indicate what sort of capabilities they', 'provide.', 'See the `packaging guides', '<https://packaging.python.org/specifications/entry-points/>`_', 'and `setuptools docs', '<https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_', 'for more information.'], 'propertyNames': {'format': 'python-entrypoint-name'}, 'additionalProperties': False, 'patternProperties': {'^.+$': {'type': 'string', '$$description': ['Reference to a Python object. It is either in the form', '``importable.module``, or ``importable.module:object.attr``.'], 'format': 'python-entrypoint-reference', '$comment': 'https://packaging.python.org/specifications/entry-points/'}}}, rule='propertyNames')
    return data