from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
@classmethod
def _generic_json_schema(cls):
    return {'type': 'object', 'properties': {'@class': {'enum': [cls.__name__], 'type': 'string'}, '@module': {'enum': [cls.__module__], 'type': 'string'}, '@version': {'type': 'string'}}, 'required': ['@class', '@module']}