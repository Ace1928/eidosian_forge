from typing import cast
from twisted.web import http
from twisted.web.iweb import IRenderable, IRequest
from twisted.web.resource import IResource, Resource
from twisted.web.template import renderElement, tags

        Handle all requests for which L{_ErrorPage} lacks a child by returning
        this error page.

        @param path: A path segment.

        @param request: HTTP request
        