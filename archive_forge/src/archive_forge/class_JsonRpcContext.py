from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
class JsonRpcContext:

    def __init__(self, entrypoint: 'Entrypoint', raw_request: Any, http_request: Request, background_tasks: BackgroundTasks, http_response: Response, json_rpc_request_class: Type[JsonRpcRequest]=JsonRpcRequest, method_route: Optional['MethodRoute']=None):
        self.entrypoint: Entrypoint = entrypoint
        self.raw_request: Any = raw_request
        self.http_request: Request = http_request
        self.background_tasks: BackgroundTasks = background_tasks
        self.http_response: Response = http_response
        self.request_class: Type[JsonRpcRequest] = json_rpc_request_class
        self.method_route: Optional[MethodRoute] = method_route
        self._raw_response: Optional[dict] = None
        self.exception: Optional[Exception] = None
        self.is_unhandled_exception: bool = False
        self.exit_stack: Optional[AsyncExitStack] = None
        self.jsonrpc_context_token: Optional[contextvars.Token] = None

    def on_raw_response(self, raw_response: dict, exception: Optional[Exception]=None, is_unhandled_exception: bool=False):
        raw_response.pop('id', None)
        if isinstance(self.raw_request, dict) and 'id' in self.raw_request:
            raw_response['id'] = self.raw_request.get('id')
        elif 'error' in raw_response:
            raw_response['id'] = None
        self._raw_response = raw_response
        self.exception = exception
        self.is_unhandled_exception = is_unhandled_exception

    @property
    def raw_response(self) -> dict:
        return self._raw_response

    @raw_response.setter
    def raw_response(self, value: dict):
        self.on_raw_response(value)

    @cached_property
    def request(self) -> JsonRpcRequest:
        try:
            return self.request_class.validate(self.raw_request)
        except DictError:
            raise InvalidRequest(data={'errors': [{'loc': (), 'type': 'type_error.dict', 'msg': 'value is not a valid dict'}]})
        except ValidationError as exc:
            raise invalid_request_from_validation_error(exc)

    async def __aenter__(self):
        assert self.exit_stack is None
        self.exit_stack = await AsyncExitStack().__aenter__()
        if sentry_sdk is not None:
            self.exit_stack.enter_context(self._fix_sentry_scope())
        await self.exit_stack.enter_async_context(self._handle_exception(reraise=False))
        self.jsonrpc_context_token = _jsonrpc_context.set(self)
        return self

    async def __aexit__(self, *exc_details):
        assert self.jsonrpc_context_token is not None
        _jsonrpc_context.reset(self.jsonrpc_context_token)
        return await self.exit_stack.__aexit__(*exc_details)

    @asynccontextmanager
    async def _handle_exception(self, reraise=True):
        try:
            yield
        except Exception as exc:
            if exc is not self.exception:
                try:
                    resp, is_unhandled_exception = await self.entrypoint.handle_exception_to_resp(exc, log_unhandled_exception=False)
                except Exception as exc:
                    self._raw_response = None
                    self.exception = exc
                    self.is_unhandled_exception = True
                    raise
                self.on_raw_response(resp, exc, is_unhandled_exception)
            if reraise:
                raise
        if self.exception is not None and self.is_unhandled_exception:
            logger.exception(str(self.exception), exc_info=self.exception)

    def _make_sentry_event_processor(self):

        def event_processor(event, _):
            if self.method_route is not None:
                event['transaction'] = sentry_transaction_from_function(self.method_route.func)
            return event
        return event_processor

    @contextmanager
    def _fix_sentry_scope(self):
        hub = sentry_sdk.Hub.current
        with sentry_sdk.Hub(hub) as hub:
            with hub.configure_scope() as scope:
                scope.clear_breadcrumbs()
                scope.add_event_processor(self._make_sentry_event_processor())
                yield

    async def enter_middlewares(self, middlewares: Sequence['JsonRpcMiddleware']):
        for mw in middlewares:
            cm = mw(self)
            if not isinstance(cm, AbstractAsyncContextManager):
                raise RuntimeError('JsonRpcMiddleware(context) must return AsyncContextManager')
            await self.exit_stack.enter_async_context(cm)
            await self.exit_stack.enter_async_context(self._handle_exception())