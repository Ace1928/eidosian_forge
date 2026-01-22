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
def _decode_dataclass(cls, kvs, infer_missing):
    if _isinstance_safe(kvs, cls):
        return kvs
    overrides = _user_overrides_or_exts(cls)
    kvs = {} if kvs is None and infer_missing else kvs
    field_names = [field.name for field in fields(cls)]
    decode_names = _decode_letter_case_overrides(field_names, overrides)
    kvs = {decode_names.get(k, k): v for k, v in kvs.items()}
    missing_fields = {field for field in fields(cls) if field.name not in kvs}
    for field in missing_fields:
        if field.default is not MISSING:
            kvs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kvs[field.name] = field.default_factory()
        elif infer_missing:
            kvs[field.name] = None
    kvs = _handle_undefined_parameters_safe(cls, kvs, usage='from')
    init_kwargs = {}
    types = get_type_hints(cls)
    for field in fields(cls):
        if not field.init:
            continue
        field_value = kvs[field.name]
        field_type = types[field.name]
        if field_value is None:
            if not _is_optional(field_type):
                warning = f'value of non-optional type {field.name} detected when decoding {cls.__name__}'
                if infer_missing:
                    warnings.warn(f'Missing {warning} and was defaulted to None by infer_missing=True. Set infer_missing=False (the default) to prevent this behavior.', RuntimeWarning)
                else:
                    warnings.warn(f"'NoneType' object {warning}.", RuntimeWarning)
            init_kwargs[field.name] = field_value
            continue
        while True:
            if not _is_new_type(field_type):
                break
            field_type = field_type.__supertype__
        if field.name in overrides and overrides[field.name].decoder is not None:
            if field_type is type(field_value):
                init_kwargs[field.name] = field_value
            else:
                init_kwargs[field.name] = overrides[field.name].decoder(field_value)
        elif is_dataclass(field_type):
            if is_dataclass(field_value):
                value = field_value
            else:
                value = _decode_dataclass(field_type, field_value, infer_missing)
            init_kwargs[field.name] = value
        elif _is_supported_generic(field_type) and field_type != str:
            init_kwargs[field.name] = _decode_generic(field_type, field_value, infer_missing)
        else:
            init_kwargs[field.name] = _support_extended_types(field_type, field_value)
    return cls(**init_kwargs)