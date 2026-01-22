import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
def _fake_send(session, request, **kwargs):
    with threading_rlock(timeout=10):
        _set_method(session, 'get_adapter', _fake_get_adapter)
        try:
            return _original_send(session, request, **kwargs)
        except exceptions.NoMockAddress:
            if not self.real_http:
                raise
        except adapter._RunRealHTTP:
            pass
        finally:
            _set_method(session, 'get_adapter', self._last_get_adapter)
    if isinstance(self._mock_target, type):
        return self._last_send(session, request, **kwargs)
    else:
        return self._last_send(request, **kwargs)