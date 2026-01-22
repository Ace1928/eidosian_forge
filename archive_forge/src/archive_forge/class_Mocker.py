import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
class Mocker(MockerCore):
    """The standard entry point for mock Adapter loading.
    """
    TEST_PREFIX = 'test'

    def __init__(self, **kwargs):
        """Create a new mocker adapter.

        :param str kw: Pass the mock object through to the decorated function
            as this named keyword argument, rather than a positional argument.
        :param bool real_http: True to send the request to the real requested
            uri if there is not a mock installed for it. Defaults to False.
        """
        self._kw = kwargs.pop('kw', None)
        super(Mocker, self).__init__(**kwargs)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def __call__(self, obj):
        if isinstance(obj, type):
            return self.decorate_class(obj)
        return self.decorate_callable(obj)

    def copy(self):
        """Returns an exact copy of current mock
        """
        m = type(self)(kw=self._kw, real_http=self.real_http, case_sensitive=self.case_sensitive)
        return m

    def decorate_callable(self, func):
        """Decorates a callable

        :param callable func: callable to decorate
        """

        @functools.wraps(func)
        def inner(*args, **kwargs):
            with self.copy() as m:
                if self._kw:
                    kwargs[self._kw] = m
                else:
                    args = list(args)
                    args.append(m)
                return func(*args, **kwargs)
        return inner

    def decorate_class(self, klass):
        """Decorates methods in a class with request_mock

        Method will be decorated only if it name begins with `TEST_PREFIX`

        :param object klass: class which methods will be decorated
        """
        for attr_name in dir(klass):
            if not attr_name.startswith(self.TEST_PREFIX):
                continue
            attr = getattr(klass, attr_name)
            if not hasattr(attr, '__call__'):
                continue
            m = self.copy()
            setattr(klass, attr_name, m(attr))
        return klass