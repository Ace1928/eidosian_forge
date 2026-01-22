import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
def _match_method(self, request):
    if self._method is ANY:
        return True
    if request.method.lower() == self._method.lower():
        return True
    return False