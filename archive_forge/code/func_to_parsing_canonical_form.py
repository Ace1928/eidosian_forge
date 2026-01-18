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
def to_parsing_canonical_form(schema: Schema) -> str:
    """Returns a string represening the parsing canonical form of the schema.

    For more details on the parsing canonical form, see here:
    https://avro.apache.org/docs/current/spec.html#Parsing+Canonical+Form+for+Schemas

    Parameters
    ----------
    schema
        Schema to transform

    """
    fo = StringIO()
    _to_parsing_canonical_form(parse_schema(schema), fo)
    return fo.getvalue()