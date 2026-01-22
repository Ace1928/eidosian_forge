import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
def add_matcher(self, matcher):
    """Register a custom matcher.

        A matcher is a callable that takes a `requests.Request` and returns a
        `requests.Response` if it matches or None if not.

        :param callable matcher: The matcher to execute.
        """
    self._matchers.append(matcher)