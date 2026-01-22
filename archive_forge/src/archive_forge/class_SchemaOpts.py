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
class SchemaOpts:
    """class Meta options for the :class:`Schema`. Defines defaults."""

    def __init__(self, meta, ordered: bool=False):
        self.fields = getattr(meta, 'fields', ())
        if not isinstance(self.fields, (list, tuple)):
            raise ValueError('`fields` option must be a list or tuple.')
        self.additional = getattr(meta, 'additional', ())
        if not isinstance(self.additional, (list, tuple)):
            raise ValueError('`additional` option must be a list or tuple.')
        if self.fields and self.additional:
            raise ValueError('Cannot set both `fields` and `additional` options for the same Schema.')
        self.exclude = getattr(meta, 'exclude', ())
        if not isinstance(self.exclude, (list, tuple)):
            raise ValueError('`exclude` must be a list or tuple.')
        self.dateformat = getattr(meta, 'dateformat', None)
        self.datetimeformat = getattr(meta, 'datetimeformat', None)
        self.timeformat = getattr(meta, 'timeformat', None)
        if hasattr(meta, 'json_module'):
            warnings.warn('The json_module class Meta option is deprecated. Use render_module instead.', RemovedInMarshmallow4Warning, stacklevel=2)
            render_module = getattr(meta, 'json_module', json)
        else:
            render_module = json
        self.render_module = getattr(meta, 'render_module', render_module)
        self.ordered = getattr(meta, 'ordered', ordered)
        self.index_errors = getattr(meta, 'index_errors', True)
        self.include = getattr(meta, 'include', {})
        self.load_only = getattr(meta, 'load_only', ())
        self.dump_only = getattr(meta, 'dump_only', ())
        self.unknown = validate_unknown_parameter_value(getattr(meta, 'unknown', RAISE))
        self.register = getattr(meta, 'register', True)