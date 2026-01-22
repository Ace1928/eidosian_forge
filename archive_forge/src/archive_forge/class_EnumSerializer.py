import builtins
import collections.abc
import datetime
import decimal
import enum
import functools
import math
import os
import pathlib
import re
import types
import uuid
from django.conf import SettingsReference
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
from django.utils.functional import LazyObject, Promise
from django.utils.version import PY311, get_docs_version
class EnumSerializer(BaseSerializer):

    def serialize(self):
        enum_class = self.value.__class__
        module = enum_class.__module__
        if issubclass(enum_class, enum.Flag):
            if PY311:
                members = list(self.value)
            else:
                members, _ = enum._decompose(enum_class, self.value)
                members = reversed(members)
        else:
            members = (self.value,)
        return (' | '.join([f'{module}.{enum_class.__qualname__}[{item.name!r}]' for item in members]), {'import %s' % module})