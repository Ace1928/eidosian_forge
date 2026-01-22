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
def _invoke_field_validators(self, *, error_store: ErrorStore, data, many: bool):
    for attr_name in self._hooks[VALIDATES]:
        validator = getattr(self, attr_name)
        validator_kwargs = validator.__marshmallow_hook__[VALIDATES]
        field_name = validator_kwargs['field_name']
        try:
            field_obj = self.fields[field_name]
        except KeyError as error:
            if field_name in self.declared_fields:
                continue
            raise ValueError(f'"{field_name}" field does not exist.') from error
        data_key = field_obj.data_key if field_obj.data_key is not None else field_name
        if many:
            for idx, item in enumerate(data):
                try:
                    value = item[field_obj.attribute or field_name]
                except KeyError:
                    pass
                else:
                    validated_value = self._call_and_store(getter_func=validator, data=value, field_name=data_key, error_store=error_store, index=idx if self.opts.index_errors else None)
                    if validated_value is missing:
                        data[idx].pop(field_name, None)
        else:
            try:
                value = data[field_obj.attribute or field_name]
            except KeyError:
                pass
            else:
                validated_value = self._call_and_store(getter_func=validator, data=value, field_name=data_key, error_store=error_store)
                if validated_value is missing:
                    data.pop(field_name, None)