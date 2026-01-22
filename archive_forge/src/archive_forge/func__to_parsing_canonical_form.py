import hashlib
from io import StringIO
import math
from os import path
from copy import deepcopy
import re
from typing import Tuple, Set, Optional, List, Any
from .types import DictSchema, Schema, NamedSchemas
from .repository import (
from .const import AVRO_TYPES
from ._schema_common import (
def _to_parsing_canonical_form(schema, fo):
    if isinstance(schema, list):
        fo.write('[')
        for idx, s in enumerate(schema):
            if idx != 0:
                fo.write(',')
            _to_parsing_canonical_form(s, fo)
        fo.write(']')
    elif not isinstance(schema, dict):
        fo.write(f'"{schema}"')
    else:
        schema_type = schema['type']
        if schema_type == 'array':
            fo.write(f'{{"type":"{schema_type}","items":')
            _to_parsing_canonical_form(schema['items'], fo)
            fo.write('}')
        elif schema_type == 'map':
            fo.write(f'{{"type":"{schema_type}","values":')
            _to_parsing_canonical_form(schema['values'], fo)
            fo.write('}')
        elif schema_type == 'enum':
            name = schema['name']
            fo.write(f'{{"name":"{name}","type":"{schema_type}","symbols":[')
            for idx, symbol in enumerate(schema['symbols']):
                if idx != 0:
                    fo.write(',')
                fo.write(f'"{symbol}"')
            fo.write(']}')
        elif schema_type == 'fixed':
            name = schema['name']
            size = schema['size']
            fo.write(f'{{"name":"{name}","type":"{schema_type}","size":{size}}}')
        elif schema_type == 'record' or schema_type == 'error':
            name = schema['name']
            fo.write(f'{{"name":"{name}","type":"record","fields":[')
            for idx, field in enumerate(schema['fields']):
                if idx != 0:
                    fo.write(',')
                name = field['name']
                fo.write(f'{{"name":"{name}","type":')
                _to_parsing_canonical_form(field['type'], fo)
                fo.write('}')
            fo.write(']}')
        elif schema_type in PRIMITIVES:
            fo.write(f'"{schema_type}"')