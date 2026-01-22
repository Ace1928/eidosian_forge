import datetime
import uuid
from hashlib import md5
import random
from string import ascii_letters
from typing import Any, Iterator, Dict, List, cast
from .const import (
from .schema import extract_record_type, extract_logical_type, parse_schema
from .types import Schema, NamedSchemas
from ._schema_common import PRIMITIVES
def _anonymize_schema(schema: Schema, named_schemas: NamedSchemas) -> Schema:
    if isinstance(schema, list):
        return [_anonymize_schema(s, named_schemas) for s in schema]
    elif not isinstance(schema, dict):
        if schema in PRIMITIVES:
            return schema
        else:
            return f'A_{_md5(schema)}'
    else:
        schema_type = schema['type']
        parsed_schema = {}
        parsed_schema['type'] = schema_type
        if 'doc' in schema:
            parsed_schema['doc'] = _md5(schema['doc'])
        if schema_type == 'array':
            parsed_schema['items'] = _anonymize_schema(schema['items'], named_schemas)
        elif schema_type == 'map':
            parsed_schema['values'] = _anonymize_schema(schema['values'], named_schemas)
        elif schema_type == 'enum':
            parsed_schema['name'] = f'A_{_md5(schema['name'])}'
            parsed_schema['symbols'] = [f'A_{_md5(symbol)}' for symbol in schema['symbols']]
        elif schema_type == 'fixed':
            parsed_schema['name'] = f'A_{_md5(schema['name'])}'
            parsed_schema['size'] = schema['size']
        elif schema_type == 'record' or schema_type == 'error':
            parsed_schema['name'] = f'A_{_md5(schema['name'])}'
            parsed_schema['fields'] = [anonymize_field(field, named_schemas) for field in schema['fields']]
        elif schema_type in PRIMITIVES:
            parsed_schema['type'] = schema_type
        return parsed_schema