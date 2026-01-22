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
def _nested_normalized_option(self, option_name: str) -> list[str]:
    nested_field = '%s.' % self.name
    return [field.split(nested_field, 1)[1] for field in getattr(self.root, option_name, set()) if field.startswith(nested_field)]