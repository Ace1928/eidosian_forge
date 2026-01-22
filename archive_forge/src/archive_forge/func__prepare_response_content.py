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
def _prepare_response_content(res: Any, *, exclude_unset: bool, exclude_defaults: bool=False, exclude_none: bool=False) -> Any:
    if isinstance(res, BaseModel):
        read_with_orm_mode = getattr(_get_model_config(res), 'read_with_orm_mode', None)
        if read_with_orm_mode:
            return res
        return _model_dump(res, by_alias=True, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)
    elif isinstance(res, list):
        return [_prepare_response_content(item, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none) for item in res]
    elif isinstance(res, dict):
        return {k: _prepare_response_content(v, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none) for k, v in res.items()}
    elif dataclasses.is_dataclass(res):
        return dataclasses.asdict(res)
    return res