import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___setuptools_pypa_io_en_latest_references_keywords_html__definitions_package_name(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, str):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be string', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/package-name', 'title': 'Valid package name', 'description': 'Valid package name (importable or PEP 561).', 'type': 'string', 'anyOf': [{'format': 'python-module-name'}, {'format': 'pep561-stub-name'}]}, rule='type')
    data_any_of_count8 = 0
    if not data_any_of_count8:
        try:
            if isinstance(data, str):
                if not custom_formats['python-module-name'](data):
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be python-module-name', value=data, name='' + (name_prefix or 'data') + '', definition={'format': 'python-module-name'}, rule='format')
            data_any_of_count8 += 1
        except JsonSchemaValueException:
            pass
    if not data_any_of_count8:
        try:
            if isinstance(data, str):
                if not custom_formats['pep561-stub-name'](data):
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be pep561-stub-name', value=data, name='' + (name_prefix or 'data') + '', definition={'format': 'pep561-stub-name'}, rule='format')
            data_any_of_count8 += 1
        except JsonSchemaValueException:
            pass
    if not data_any_of_count8:
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' cannot be validated by any definition', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/package-name', 'title': 'Valid package name', 'description': 'Valid package name (importable or PEP 561).', 'type': 'string', 'anyOf': [{'format': 'python-module-name'}, {'format': 'pep561-stub-name'}]}, rule='anyOf')
    return data