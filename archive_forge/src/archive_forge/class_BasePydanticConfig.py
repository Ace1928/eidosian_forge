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
class BasePydanticConfig:
    orm_mode = True
    arbitrary_types_allowed = True
    json_encoders = _json_encoders
    json_loads = Json.loads
    json_dumps = Json.dumps
    extra = 'allow'