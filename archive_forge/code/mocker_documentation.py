import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
Decorates methods in a class with request_mock

        Method will be decorated only if it name begins with `TEST_PREFIX`

        :param object klass: class which methods will be decorated
        