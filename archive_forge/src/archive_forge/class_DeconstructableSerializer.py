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
class DeconstructableSerializer(BaseSerializer):

    @staticmethod
    def serialize_deconstructed(path, args, kwargs):
        name, imports = DeconstructableSerializer._serialize_path(path)
        strings = []
        for arg in args:
            arg_string, arg_imports = serializer_factory(arg).serialize()
            strings.append(arg_string)
            imports.update(arg_imports)
        for kw, arg in sorted(kwargs.items()):
            arg_string, arg_imports = serializer_factory(arg).serialize()
            imports.update(arg_imports)
            strings.append('%s=%s' % (kw, arg_string))
        return ('%s(%s)' % (name, ', '.join(strings)), imports)

    @staticmethod
    def _serialize_path(path):
        module, name = path.rsplit('.', 1)
        if module == 'django.db.models':
            imports = {'from django.db import models'}
            name = 'models.%s' % name
        else:
            imports = {'import %s' % module}
            name = path
        return (name, imports)

    def serialize(self):
        return self.serialize_deconstructed(*self.value.deconstruct())