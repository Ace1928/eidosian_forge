import asyncio
import dataclasses
import email.message
import inspect
import json
from contextlib import AsyncExitStack
from enum import Enum, IntEnum
from typing import (
from fastapi import params
from fastapi._compat import (
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import (
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import (
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import (
from pydantic import BaseModel
from starlette import routing
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import (
from starlette.routing import Mount as Mount  # noqa
from starlette.types import ASGIApp, Lifespan, Scope
from starlette.websockets import WebSocket
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def api_route(self, path: str, *, response_model: Any=Default(None), status_code: Optional[int]=None, tags: Optional[List[Union[str, Enum]]]=None, dependencies: Optional[Sequence[params.Depends]]=None, summary: Optional[str]=None, description: Optional[str]=None, response_description: str='Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]]=None, deprecated: Optional[bool]=None, methods: Optional[List[str]]=None, operation_id: Optional[str]=None, response_model_include: Optional[IncEx]=None, response_model_exclude: Optional[IncEx]=None, response_model_by_alias: bool=True, response_model_exclude_unset: bool=False, response_model_exclude_defaults: bool=False, response_model_exclude_none: bool=False, include_in_schema: bool=True, response_class: Type[Response]=Default(JSONResponse), name: Optional[str]=None, callbacks: Optional[List[BaseRoute]]=None, openapi_extra: Optional[Dict[str, Any]]=None, generate_unique_id_function: Callable[[APIRoute], str]=Default(generate_unique_id)) -> Callable[[DecoratedCallable], DecoratedCallable]:

    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        self.add_api_route(path, func, response_model=response_model, status_code=status_code, tags=tags, dependencies=dependencies, summary=summary, description=description, response_description=response_description, responses=responses, deprecated=deprecated, methods=methods, operation_id=operation_id, response_model_include=response_model_include, response_model_exclude=response_model_exclude, response_model_by_alias=response_model_by_alias, response_model_exclude_unset=response_model_exclude_unset, response_model_exclude_defaults=response_model_exclude_defaults, response_model_exclude_none=response_model_exclude_none, include_in_schema=include_in_schema, response_class=response_class, name=name, callbacks=callbacks, openapi_extra=openapi_extra, generate_unique_id_function=generate_unique_id_function)
        return func
    return decorator