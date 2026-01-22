import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
def _match_additional(self, request):
    if callable(self._additional_matcher):
        return self._additional_matcher(request)
    if self._additional_matcher is not None:
        raise TypeError('Unexpected format of additional matcher.')
    return True