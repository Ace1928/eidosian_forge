from uvicorn._types import (
class ASGI2Middleware:

    def __init__(self, app: 'ASGI2Application'):
        self.app = app

    async def __call__(self, scope: 'Scope', receive: 'ASGIReceiveCallable', send: 'ASGISendCallable') -> None:
        instance = self.app(scope)
        await instance(receive, send)