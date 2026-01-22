from __future__ import annotations
import collections
import copy
import datetime as dt
import decimal
import ipaddress
import math
import numbers
import typing
import uuid
import warnings
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType
from marshmallow import class_registry, types, utils, validate
from marshmallow.base import FieldABC, SchemaABC
from marshmallow.exceptions import (
from marshmallow.utils import (
from marshmallow.utils import (
from marshmallow.validate import And, Length
from marshmallow.warnings import RemovedInMarshmallow4Warning
def _validate_missing(self, value):
    """Validate missing values. Raise a :exc:`ValidationError` if
        `value` should be considered missing.
        """
    if value is missing_ and self.required:
        raise self.make_error('required')
    if value is None and (not self.allow_none):
        raise self.make_error('null')