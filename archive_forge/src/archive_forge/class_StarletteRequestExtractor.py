from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
class StarletteRequestExtractor:
    """
    Extracts useful information from the Starlette request
    (like form data or cookies) and adds it to the Sentry event.
    """
    request = None

    def __init__(self, request):
        self.request = request

    def extract_cookies_from_request(self):
        client = Hub.current.client
        if client is None:
            return None
        cookies = None
        if _should_send_default_pii():
            cookies = self.cookies()
        return cookies

    async def extract_request_info(self):
        client = Hub.current.client
        if client is None:
            return None
        request_info = {}
        with capture_internal_exceptions():
            if _should_send_default_pii():
                request_info['cookies'] = self.cookies()
            content_length = await self.content_length()
            if not content_length:
                return request_info
            if content_length and (not request_body_within_bounds(client, content_length)):
                request_info['data'] = AnnotatedValue.removed_because_over_size_limit()
                return request_info
            json = await self.json()
            if json:
                request_info['data'] = json
                return request_info
            form = await self.form()
            if form:
                form_data = {}
                for key, val in iteritems(form):
                    is_file = isinstance(val, UploadFile)
                    form_data[key] = val if not is_file else AnnotatedValue.removed_because_raw_data()
                request_info['data'] = form_data
                return request_info
            request_info['data'] = AnnotatedValue.removed_because_raw_data()
            return request_info

    async def content_length(self):
        if 'content-length' in self.request.headers:
            return int(self.request.headers['content-length'])
        return None

    def cookies(self):
        return self.request.cookies

    async def form(self):
        if multipart is None:
            return None
        await self.request.body()
        return await self.request.form()

    def is_json(self):
        return _is_json_content_type(self.request.headers.get('content-type'))

    async def json(self):
        if not self.is_json():
            return None
        return await self.request.json()