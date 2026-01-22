from __future__ import annotations
from contextlib import contextmanager
from ansible.module_utils.basic import missing_required_lib
from .vendor.hcloud import APIException, Client as ClientBase
class CachedSession(requests.Session):
    cache: dict[str, requests.Response]

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        """
            Send a given PreparedRequest.
            """
        if request.method != 'GET' or request.url is None:
            return super().send(request, **kwargs)
        if request.url in self.cache:
            return self.cache[request.url]
        response = super().send(request, **kwargs)
        if response.ok:
            self.cache[request.url] = response
        return response