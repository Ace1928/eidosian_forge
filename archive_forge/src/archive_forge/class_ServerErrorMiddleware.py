import html
import inspect
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
class ServerErrorMiddleware:
    """
    Handles returning 500 responses when a server error occurs.

    If 'debug' is set, then traceback responses will be returned,
    otherwise the designated 'handler' will be called.

    This middleware class should generally be used to wrap *everything*
    else up, so that unhandled exceptions anywhere in the stack
    always result in an appropriate 500 response.
    """

    def __init__(self, app: ASGIApp, handler: typing.Optional[typing.Callable[[Request, Exception], typing.Any]]=None, debug: bool=False) -> None:
        self.app = app
        self.handler = handler
        self.debug = debug

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        response_started = False

        async def _send(message: Message) -> None:
            nonlocal response_started, send
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)
        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            request = Request(scope)
            if self.debug:
                response = self.debug_response(request, exc)
            elif self.handler is None:
                response = self.error_response(request, exc)
            elif is_async_callable(self.handler):
                response = await self.handler(request, exc)
            else:
                response = await run_in_threadpool(self.handler, request, exc)
            if not response_started:
                await response(scope, receive, send)
            raise exc

    def format_line(self, index: int, line: str, frame_lineno: int, frame_index: int) -> str:
        values = {'line': html.escape(line).replace(' ', '&nbsp'), 'lineno': frame_lineno - frame_index + index}
        if index != frame_index:
            return LINE.format(**values)
        return CENTER_LINE.format(**values)

    def generate_frame_html(self, frame: inspect.FrameInfo, is_collapsed: bool) -> str:
        code_context = ''.join((self.format_line(index, line, frame.lineno, frame.index) for index, line in enumerate(frame.code_context or [])))
        values = {'frame_filename': html.escape(frame.filename), 'frame_lineno': frame.lineno, 'frame_name': html.escape(frame.function), 'code_context': code_context, 'collapsed': 'collapsed' if is_collapsed else '', 'collapse_button': '+' if is_collapsed else '&#8210;'}
        return FRAME_TEMPLATE.format(**values)

    def generate_html(self, exc: Exception, limit: int=7) -> str:
        traceback_obj = traceback.TracebackException.from_exception(exc, capture_locals=True)
        exc_html = ''
        is_collapsed = False
        exc_traceback = exc.__traceback__
        if exc_traceback is not None:
            frames = inspect.getinnerframes(exc_traceback, limit)
            for frame in reversed(frames):
                exc_html += self.generate_frame_html(frame, is_collapsed)
                is_collapsed = True
        error = f'{html.escape(traceback_obj.exc_type.__name__)}: {html.escape(str(traceback_obj))}'
        return TEMPLATE.format(styles=STYLES, js=JS, error=error, exc_html=exc_html)

    def generate_plain_text(self, exc: Exception) -> str:
        return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def debug_response(self, request: Request, exc: Exception) -> Response:
        accept = request.headers.get('accept', '')
        if 'text/html' in accept:
            content = self.generate_html(exc)
            return HTMLResponse(content, status_code=500)
        content = self.generate_plain_text(exc)
        return PlainTextResponse(content, status_code=500)

    def error_response(self, request: Request, exc: Exception) -> Response:
        return PlainTextResponse('Internal Server Error', status_code=500)