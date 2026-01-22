import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class HTTPGitApplication:
    """Class encapsulating the state of a git WSGI application.

    Attributes:
      backend: the Backend object backing this application
    """
    services: ClassVar[Dict[Tuple[str, re.Pattern], Callable[[HTTPGitRequest, Backend, re.Match], Iterator[bytes]]]] = {('GET', re.compile('/HEAD$')): get_text_file, ('GET', re.compile('/info/refs$')): get_info_refs, ('GET', re.compile('/objects/info/alternates$')): get_text_file, ('GET', re.compile('/objects/info/http-alternates$')): get_text_file, ('GET', re.compile('/objects/info/packs$')): get_info_packs, ('GET', re.compile('/objects/([0-9a-f]{2})/([0-9a-f]{38})$')): get_loose_object, ('GET', re.compile('/objects/pack/pack-([0-9a-f]{40})\\.pack$')): get_pack_file, ('GET', re.compile('/objects/pack/pack-([0-9a-f]{40})\\.idx$')): get_idx_file, ('POST', re.compile('/git-upload-pack$')): handle_service_request, ('POST', re.compile('/git-receive-pack$')): handle_service_request}

    def __init__(self, backend, dumb: bool=False, handlers=None, fallback_app=None) -> None:
        self.backend = backend
        self.dumb = dumb
        self.handlers = dict(DEFAULT_HANDLERS)
        self.fallback_app = fallback_app
        if handlers is not None:
            self.handlers.update(handlers)

    def __call__(self, environ, start_response):
        path = environ['PATH_INFO']
        method = environ['REQUEST_METHOD']
        req = HTTPGitRequest(environ, start_response, dumb=self.dumb, handlers=self.handlers)
        handler = None
        for smethod, spath in self.services.keys():
            if smethod != method:
                continue
            mat = spath.search(path)
            if mat:
                handler = self.services[smethod, spath]
                break
        if handler is None:
            if self.fallback_app is not None:
                return self.fallback_app(environ, start_response)
            else:
                return [req.not_found('Sorry, that method is not supported')]
        return handler(req, self.backend, mat)