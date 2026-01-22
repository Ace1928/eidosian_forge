from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
class _StreamingFetcher:

    def __init__(self) -> None:
        self.streaming_ready = False
        js_data_blob = js.Blob.new([_STREAMING_WORKER_CODE], _obj_from_dict({'type': 'application/javascript'}))

        def promise_resolver(js_resolve_fn: JsProxy, js_reject_fn: JsProxy) -> None:

            def onMsg(e: JsProxy) -> None:
                self.streaming_ready = True
                js_resolve_fn(e)

            def onErr(e: JsProxy) -> None:
                js_reject_fn(e)
            self.js_worker.onmessage = onMsg
            self.js_worker.onerror = onErr
        js_data_url = js.URL.createObjectURL(js_data_blob)
        self.js_worker = js.globalThis.Worker.new(js_data_url)
        self.js_worker_ready_promise = js.globalThis.Promise.new(promise_resolver)

    def send(self, request: EmscriptenRequest) -> EmscriptenResponse:
        headers = {k: v for k, v in request.headers.items() if k not in HEADERS_TO_IGNORE}
        body = request.body
        fetch_data = {'headers': headers, 'body': to_js(body), 'method': request.method}
        timeout = int(1000 * request.timeout) if request.timeout > 0 else None
        js_shared_buffer = js.SharedArrayBuffer.new(1048576)
        js_int_buffer = js.Int32Array.new(js_shared_buffer)
        js_byte_buffer = js.Uint8Array.new(js_shared_buffer, 8)
        js.Atomics.store(js_int_buffer, 0, ERROR_TIMEOUT)
        js.Atomics.notify(js_int_buffer, 0)
        js_absolute_url = js.URL.new(request.url, js.location).href
        self.js_worker.postMessage(_obj_from_dict({'buffer': js_shared_buffer, 'url': js_absolute_url, 'fetchParams': fetch_data}))
        js.Atomics.wait(js_int_buffer, 0, ERROR_TIMEOUT, timeout)
        if js_int_buffer[0] == ERROR_TIMEOUT:
            raise _TimeoutError('Timeout connecting to streaming request', request=request, response=None)
        elif js_int_buffer[0] == SUCCESS_HEADER:
            string_len = js_int_buffer[1]
            js_decoder = js.TextDecoder.new()
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            response_obj = json.loads(json_str)
            return EmscriptenResponse(request=request, status_code=response_obj['status'], headers=response_obj['headers'], body=_ReadStream(js_int_buffer, js_byte_buffer, request.timeout, self.js_worker, response_obj['connectionID'], request))
        elif js_int_buffer[0] == ERROR_EXCEPTION:
            string_len = js_int_buffer[1]
            js_decoder = js.TextDecoder.new()
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            raise _StreamingError(f'Exception thrown in fetch: {json_str}', request=request, response=None)
        else:
            raise _StreamingError(f'Unknown status from worker in fetch: {js_int_buffer[0]}', request=request, response=None)