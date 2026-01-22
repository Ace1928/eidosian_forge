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
class APIRoute(routing.Route):

    def __init__(self, path: str, endpoint: Callable[..., Any], *, response_model: Any=Default(None), status_code: Optional[int]=None, tags: Optional[List[Union[str, Enum]]]=None, dependencies: Optional[Sequence[params.Depends]]=None, summary: Optional[str]=None, description: Optional[str]=None, response_description: str='Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]]=None, deprecated: Optional[bool]=None, name: Optional[str]=None, methods: Optional[Union[Set[str], List[str]]]=None, operation_id: Optional[str]=None, response_model_include: Optional[IncEx]=None, response_model_exclude: Optional[IncEx]=None, response_model_by_alias: bool=True, response_model_exclude_unset: bool=False, response_model_exclude_defaults: bool=False, response_model_exclude_none: bool=False, include_in_schema: bool=True, response_class: Union[Type[Response], DefaultPlaceholder]=Default(JSONResponse), dependency_overrides_provider: Optional[Any]=None, callbacks: Optional[List[BaseRoute]]=None, openapi_extra: Optional[Dict[str, Any]]=None, generate_unique_id_function: Union[Callable[['APIRoute'], str], DefaultPlaceholder]=Default(generate_unique_id)) -> None:
        self.path = path
        self.endpoint = endpoint
        if isinstance(response_model, DefaultPlaceholder):
            return_annotation = get_typed_return_annotation(endpoint)
            if lenient_issubclass(return_annotation, Response):
                response_model = None
            else:
                response_model = return_annotation
        self.response_model = response_model
        self.summary = summary
        self.response_description = response_description
        self.deprecated = deprecated
        self.operation_id = operation_id
        self.response_model_include = response_model_include
        self.response_model_exclude = response_model_exclude
        self.response_model_by_alias = response_model_by_alias
        self.response_model_exclude_unset = response_model_exclude_unset
        self.response_model_exclude_defaults = response_model_exclude_defaults
        self.response_model_exclude_none = response_model_exclude_none
        self.include_in_schema = include_in_schema
        self.response_class = response_class
        self.dependency_overrides_provider = dependency_overrides_provider
        self.callbacks = callbacks
        self.openapi_extra = openapi_extra
        self.generate_unique_id_function = generate_unique_id_function
        self.tags = tags or []
        self.responses = responses or {}
        self.name = get_name(endpoint) if name is None else name
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)
        if methods is None:
            methods = ['GET']
        self.methods: Set[str] = {method.upper() for method in methods}
        if isinstance(generate_unique_id_function, DefaultPlaceholder):
            current_generate_unique_id: Callable[['APIRoute'], str] = generate_unique_id_function.value
        else:
            current_generate_unique_id = generate_unique_id_function
        self.unique_id = self.operation_id or current_generate_unique_id(self)
        if isinstance(status_code, IntEnum):
            status_code = int(status_code)
        self.status_code = status_code
        if self.response_model:
            assert is_body_allowed_for_status_code(status_code), f'Status code {status_code} must not have a response body'
            response_name = 'Response_' + self.unique_id
            self.response_field = create_response_field(name=response_name, type_=self.response_model, mode='serialization')
            self.secure_cloned_response_field: Optional[ModelField] = create_cloned_field(self.response_field)
        else:
            self.response_field = None
            self.secure_cloned_response_field = None
        self.dependencies = list(dependencies or [])
        self.description = description or inspect.cleandoc(self.endpoint.__doc__ or '')
        self.description = self.description.split('\x0c')[0].strip()
        response_fields = {}
        for additional_status_code, response in self.responses.items():
            assert isinstance(response, dict), 'An additional response must be a dict'
            model = response.get('model')
            if model:
                assert is_body_allowed_for_status_code(additional_status_code), f'Status code {additional_status_code} must not have a response body'
                response_name = f'Response_{additional_status_code}_{self.unique_id}'
                response_field = create_response_field(name=response_name, type_=model)
                response_fields[additional_status_code] = response_field
        if response_fields:
            self.response_fields: Dict[Union[int, str], ModelField] = response_fields
        else:
            self.response_fields = {}
        assert callable(endpoint), 'An endpoint must be a callable'
        self.dependant = get_dependant(path=self.path_format, call=self.endpoint)
        for depends in self.dependencies[::-1]:
            self.dependant.dependencies.insert(0, get_parameterless_sub_dependant(depends=depends, path=self.path_format))
        self.body_field = get_body_field(dependant=self.dependant, name=self.unique_id)
        self.app = request_response(self.get_route_handler())

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        return get_request_handler(dependant=self.dependant, body_field=self.body_field, status_code=self.status_code, response_class=self.response_class, response_field=self.secure_cloned_response_field, response_model_include=self.response_model_include, response_model_exclude=self.response_model_exclude, response_model_by_alias=self.response_model_by_alias, response_model_exclude_unset=self.response_model_exclude_unset, response_model_exclude_defaults=self.response_model_exclude_defaults, response_model_exclude_none=self.response_model_exclude_none, dependency_overrides_provider=self.dependency_overrides_provider)

    def matches(self, scope: Scope) -> Tuple[Match, Scope]:
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope['route'] = self
        return (match, child_scope)