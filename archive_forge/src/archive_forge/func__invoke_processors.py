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
def _invoke_processors(self, tag: str, *, pass_many: bool, data, many: bool, original_data=None, **kwargs):
    key = (tag, pass_many)
    for attr_name in self._hooks[key]:
        processor = getattr(self, attr_name)
        processor_kwargs = processor.__marshmallow_hook__[key]
        pass_original = processor_kwargs.get('pass_original', False)
        if many and (not pass_many):
            if pass_original:
                data = [processor(item, original, many=many, **kwargs) for item, original in zip(data, original_data)]
            else:
                data = [processor(item, many=many, **kwargs) for item in data]
        elif pass_original:
            data = processor(data, original_data, many=many, **kwargs)
        else:
            data = processor(data, many=many, **kwargs)
    return data