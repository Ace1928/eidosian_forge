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
def _normalize_nested_options(self) -> None:
    """Apply then flatten nested schema options.
        This method is private API.
        """
    if self.only is not None:
        self.__apply_nested_option('only', self.only, 'intersection')
        self.only = self.set_class([field.split('.', 1)[0] for field in self.only])
    if self.exclude:
        self.__apply_nested_option('exclude', self.exclude, 'union')
        self.exclude = self.set_class([field for field in self.exclude if '.' not in field])