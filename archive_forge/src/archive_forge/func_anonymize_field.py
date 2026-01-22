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
def anonymize_field(field: Dict[str, Any], named_schemas: NamedSchemas) -> Dict[str, Any]:
    parsed_field: Dict[str, Any] = {}
    if 'doc' in field:
        parsed_field['doc'] = _md5(field['doc'])
    if 'aliases' in field:
        parsed_field['aliases'] = [_md5(alias) for alias in field['aliases']]
    if 'default' in field:
        parsed_field['default'] = field['default']
    parsed_field['name'] = _md5(field['name'])
    parsed_field['type'] = _anonymize_schema(field['type'], named_schemas)
    return parsed_field