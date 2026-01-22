from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class _GoogleAPICallErrorMeta(type):
    """Metaclass for registering GoogleAPICallError subclasses."""

    def __new__(mcs, name, bases, class_dict):
        cls = type.__new__(mcs, name, bases, class_dict)
        if cls.code is not None:
            _HTTP_CODE_TO_EXCEPTION.setdefault(cls.code, cls)
        if cls.grpc_status_code is not None:
            _GRPC_CODE_TO_EXCEPTION.setdefault(cls.grpc_status_code, cls)
        return cls