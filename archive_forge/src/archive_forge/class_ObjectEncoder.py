import uuid
import json
import typing
import codecs
import hashlib
import datetime
import contextlib
import dataclasses
from enum import Enum
from .lazy import lazy_import, get_obj_class_name
class ObjectEncoder(json.JSONEncoder):

    def default(self, obj: typing.Any):
        with contextlib.suppress(Exception):
            return object_serializer(obj)
        return json.JSONEncoder.default(self, obj)