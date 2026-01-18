import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___packaging_python_org_en_latest_specifications_declaring_project_metadata___definitions_dependency(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, str):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be string', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/dependency', 'title': 'Dependency', 'type': 'string', 'description': 'Project dependency specification according to PEP 508', 'format': 'pep508'}, rule='type')
    if isinstance(data, str):
        if not custom_formats['pep508'](data):
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be pep508', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/dependency', 'title': 'Dependency', 'type': 'string', 'description': 'Project dependency specification according to PEP 508', 'format': 'pep508'}, rule='format')
    return data