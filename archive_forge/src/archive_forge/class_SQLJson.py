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
class SQLJson(Json):

    @staticmethod
    def dumps(obj: typing.Dict[typing.Any, typing.Any], *args, default: typing.Dict[typing.Any, typing.Any]=None, cls: typing.Type[json.JSONEncoder]=AlchemyEncoder, _fallback_method: typing.Optional[typing.Callable]=None, **kwargs) -> str:
        try:
            return json.dumps(obj, *args, default=default, cls=cls, **kwargs)
        except Exception as e:
            if _fallback_method is not None:
                return _fallback_method(obj, *args, default=default, **kwargs)
            raise e