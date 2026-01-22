from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
def _init_fields(self) -> None:
    """Update self.fields, self.load_fields, and self.dump_fields based on schema options.
        This method is private API.
        """
    if self.opts.fields:
        available_field_names = self.set_class(self.opts.fields)
    else:
        available_field_names = self.set_class(self.declared_fields.keys())
        if self.opts.additional:
            available_field_names |= self.set_class(self.opts.additional)
    invalid_fields = self.set_class()
    if self.only is not None:
        field_names: typing.AbstractSet[typing.Any] = self.set_class(self.only)
        invalid_fields |= field_names - available_field_names
    else:
        field_names = available_field_names
    if self.exclude:
        field_names = field_names - self.exclude
        invalid_fields |= self.exclude - available_field_names
    if invalid_fields:
        message = f'Invalid fields for {self}: {invalid_fields}.'
        raise ValueError(message)
    fields_dict = self.dict_class()
    for field_name in field_names:
        field_obj = self.declared_fields.get(field_name, ma_fields.Inferred())
        self._bind_field(field_name, field_obj)
        fields_dict[field_name] = field_obj
    load_fields, dump_fields = (self.dict_class(), self.dict_class())
    for field_name, field_obj in fields_dict.items():
        if not field_obj.dump_only:
            load_fields[field_name] = field_obj
        if not field_obj.load_only:
            dump_fields[field_name] = field_obj
    dump_data_keys = [field_obj.data_key if field_obj.data_key is not None else name for name, field_obj in dump_fields.items()]
    if len(dump_data_keys) != len(set(dump_data_keys)):
        data_keys_duplicates = {x for x in dump_data_keys if dump_data_keys.count(x) > 1}
        raise ValueError(f"The data_key argument for one or more fields collides with another field's name or data_key argument. Check the following field names and data_key arguments: {list(data_keys_duplicates)}")
    load_attributes = [obj.attribute or name for name, obj in load_fields.items()]
    if len(load_attributes) != len(set(load_attributes)):
        attributes_duplicates = {x for x in load_attributes if load_attributes.count(x) > 1}
        raise ValueError(f"The attribute argument for one or more fields collides with another field's name or attribute argument. Check the following field names and attribute arguments: {list(attributes_duplicates)}")
    self.fields = fields_dict
    self.dump_fields = dump_fields
    self.load_fields = load_fields