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
def _inject_schema(outer_schema, inner_schema, namespace='', is_injected=False):
    if is_injected is True:
        return (outer_schema, is_injected)
    if isinstance(outer_schema, list):
        union = []
        for each_schema in outer_schema:
            if is_injected:
                union.append(each_schema)
            else:
                return_schema, injected = _inject_schema(each_schema, inner_schema, namespace, is_injected)
                union.append(return_schema)
                if injected is True:
                    is_injected = injected
        return (union, is_injected)
    elif not isinstance(outer_schema, dict):
        if outer_schema in PRIMITIVES:
            return (outer_schema, is_injected)
        if '.' not in outer_schema and namespace:
            outer_schema = namespace + '.' + outer_schema
        if outer_schema == inner_schema['name']:
            return (inner_schema, True)
        else:
            return (outer_schema, is_injected)
    else:
        schema_type = outer_schema['type']
        if schema_type == 'array':
            return_schema, injected = _inject_schema(outer_schema['items'], inner_schema, namespace, is_injected)
            outer_schema['items'] = return_schema
            return (outer_schema, injected)
        elif schema_type == 'map':
            return_schema, injected = _inject_schema(outer_schema['values'], inner_schema, namespace, is_injected)
            outer_schema['values'] = return_schema
            return (outer_schema, injected)
        elif schema_type == 'enum':
            return (outer_schema, is_injected)
        elif schema_type == 'fixed':
            return (outer_schema, is_injected)
        elif schema_type == 'record' or schema_type == 'error':
            namespace, _ = schema_name(outer_schema, namespace)
            fields = []
            for field in outer_schema.get('fields', []):
                if is_injected:
                    fields.append(field)
                else:
                    return_schema, injected = _inject_schema(field['type'], inner_schema, namespace, is_injected)
                    field['type'] = return_schema
                    fields.append(field)
                    if injected is True:
                        is_injected = injected
            if fields:
                outer_schema['fields'] = fields
            return (outer_schema, is_injected)
        elif schema_type in PRIMITIVES:
            return (outer_schema, is_injected)
        else:
            raise Exception('Internal error; ' + 'You should raise an issue in the fastavro github repository')