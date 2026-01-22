import copy
import json
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import (MISSING,
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (Any, Collection, Mapping, Union, get_type_hints,
from uuid import UUID
from typing_inspect import is_union_type  # type: ignore
from dataclasses_json import cfg
from dataclasses_json.utils import (_get_type_cons, _get_type_origin,
def _user_overrides_or_exts(cls):
    global_metadata = defaultdict(dict)
    encoders = cfg.global_config.encoders
    decoders = cfg.global_config.decoders
    mm_fields = cfg.global_config.mm_fields
    for field in fields(cls):
        if field.type in encoders:
            global_metadata[field.name]['encoder'] = encoders[field.type]
        if field.type in decoders:
            global_metadata[field.name]['decoder'] = decoders[field.type]
        if field.type in mm_fields:
            global_metadata[field.name]['mm_field'] = mm_fields[field.type]
    try:
        cls_config = cls.dataclass_json_config if cls.dataclass_json_config is not None else {}
    except AttributeError:
        cls_config = {}
    overrides = {}
    for field in fields(cls):
        field_config = {}
        field_metadata = global_metadata[field.name]
        if 'encoder' in field_metadata:
            field_config['encoder'] = field_metadata['encoder']
        if 'decoder' in field_metadata:
            field_config['decoder'] = field_metadata['decoder']
        if 'mm_field' in field_metadata:
            field_config['mm_field'] = field_metadata['mm_field']
        field_config.update(cls_config)
        field_config.update(field.metadata.get('dataclasses_json', {}))
        overrides[field.name] = FieldOverride(*map(field_config.get, confs))
    return overrides