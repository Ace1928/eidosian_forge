import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
def _fake_get_adapter(session, url):
    return self._adapter