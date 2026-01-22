import json
import typing
import datetime
import contextlib
from enum import Enum
from sqlalchemy import inspect
from lazyops.utils.serialization import object_serializer, Json
from sqlalchemy.ext.declarative import DeclarativeMeta
from pydantic import create_model, BaseModel, Field
from typing import Optional, Dict, Any, List, Union, Type, cast
class AlchemyEncoder(json.JSONEncoder):

    def default(self, obj):
        if not isinstance(obj.__class__, DeclarativeMeta):
            return json.JSONEncoder.default(self, obj)
        fields = {}
        for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
            data = obj.__getattribute__(field)
            try:
                json.dumps(data)
                fields[field] = data
            except TypeError:
                fields[field] = None
        return fields