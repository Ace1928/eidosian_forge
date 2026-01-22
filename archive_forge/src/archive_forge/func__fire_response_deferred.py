import logging
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from h2.errors import ErrorCodes
from h2.exceptions import H2Error, ProtocolError, StreamClosedError
from hpack import HeaderTuple
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.error import ConnectionClosed
from twisted.python.failure import Failure
from twisted.web.client import ResponseFailed
from scrapy.http import Request
from scrapy.http.headers import Headers
from scrapy.responsetypes import responsetypes
def _fire_response_deferred(self) -> None:
    """Builds response from the self._response dict
        and fires the response deferred callback with the
        generated response instance"""
    body = self._response['body'].getvalue()
    response_cls = responsetypes.from_args(headers=self._response['headers'], url=self._request.url, body=body)
    response = response_cls(url=self._request.url, status=int(self._response['headers'][':status']), headers=self._response['headers'], body=body, request=self._request, certificate=self._protocol.metadata['certificate'], ip_address=self._protocol.metadata['ip_address'], protocol='h2')
    self._deferred_response.callback(response)