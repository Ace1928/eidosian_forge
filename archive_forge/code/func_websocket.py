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
def websocket(self, path: Annotated[str, Doc('\n                WebSocket path.\n                ')], name: Annotated[Optional[str], Doc('\n                A name for the WebSocket. Only used internally.\n                ')]=None, *, dependencies: Annotated[Optional[Sequence[params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be used for this\n                WebSocket.\n\n                Read more about it in the\n                [FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).\n                ')]=None) -> Callable[[DecoratedCallable], DecoratedCallable]:
    """
        Decorate a WebSocket function.

        Read more about it in the
        [FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).

        **Example**

        ## Example

        ```python
        from fastapi import APIRouter, FastAPI, WebSocket

        app = FastAPI()
        router = APIRouter()

        @router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Message text was: {data}")

        app.include_router(router)
        ```
        """

    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        self.add_api_websocket_route(path, func, name=name, dependencies=dependencies)
        return func
    return decorator