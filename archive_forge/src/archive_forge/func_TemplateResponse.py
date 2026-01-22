from __future__ import annotations
import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send
def TemplateResponse(self, *args: typing.Any, **kwargs: typing.Any) -> _TemplateResponse:
    if args:
        if isinstance(args[0], str):
            warnings.warn('The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.\nReplace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.', DeprecationWarning)
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get('context', {})
            status_code = args[2] if len(args) > 2 else kwargs.get('status_code', 200)
            headers = args[2] if len(args) > 2 else kwargs.get('headers')
            media_type = args[3] if len(args) > 3 else kwargs.get('media_type')
            background = args[4] if len(args) > 4 else kwargs.get('background')
            if 'request' not in context:
                raise ValueError('context must include a "request" key')
            request = context['request']
        else:
            request = args[0]
            name = args[1] if len(args) > 1 else kwargs['name']
            context = args[2] if len(args) > 2 else kwargs.get('context', {})
            status_code = args[3] if len(args) > 3 else kwargs.get('status_code', 200)
            headers = args[4] if len(args) > 4 else kwargs.get('headers')
            media_type = args[5] if len(args) > 5 else kwargs.get('media_type')
            background = args[6] if len(args) > 6 else kwargs.get('background')
    else:
        if 'request' not in kwargs:
            warnings.warn('The `TemplateResponse` now requires the `request` argument.\nReplace `TemplateResponse(name, {"context": context})` by `TemplateResponse(request, name)`.', DeprecationWarning)
            if 'request' not in kwargs.get('context', {}):
                raise ValueError('context must include a "request" key')
        context = kwargs.get('context', {})
        request = kwargs.get('request', context.get('request'))
        name = typing.cast(str, kwargs['name'])
        status_code = kwargs.get('status_code', 200)
        headers = kwargs.get('headers')
        media_type = kwargs.get('media_type')
        background = kwargs.get('background')
    context.setdefault('request', request)
    for context_processor in self.context_processors:
        context.update(context_processor(request))
    template = self.get_template(name)
    return _TemplateResponse(template, context, status_code=status_code, headers=headers, media_type=media_type, background=background)