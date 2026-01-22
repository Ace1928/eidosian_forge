from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
def _stub_for_method(self, method):
    method = _simplify_method_name(method)
    self._method_stubs[method] = _CallableStub(method, self)
    return self._method_stubs[method]