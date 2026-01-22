import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
class RequestFailure(Error):
    """
    A request failed, possibly after some number of retries
    """

    def __init__(self, message: str, request_string: str, response_status: int, error: Optional[str], error_description: Optional[str], error_headers: Optional[str]=None):
        self.request_string: str = request_string
        self.response_status: int = response_status
        self.error: Optional[str] = error
        self.error_description: Optional[str] = error_description
        self.error_headers: Optional[str] = error_headers
        super().__init__(message, self.request_string, self.response_status, self.error, self.error_description, self.error_headers)

    def __str__(self) -> str:
        return f'message={self.message}, request={self.request_string}, status={self.response_status}, error={self.error}, error_description={self.error_description}, error_headers={self.error_headers}'

    @classmethod
    def create_from_request_response(cls, message: str, request: Request, response: 'urllib3.BaseHTTPResponse') -> Any:
        err, err_desc, err_headers = _extract_error_from_response(response)
        return cls(message=message, request_string=str(request), response_status=response.status, error=err, error_description=err_desc, error_headers=err_headers)