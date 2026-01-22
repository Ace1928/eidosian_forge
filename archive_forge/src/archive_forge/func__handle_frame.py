import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
def _handle_frame(self, resp):
    correlation_id, resp_type, fut = self._requests[0]
    if correlation_id is None:
        if not fut.done():
            fut.set_result(resp)
    else:
        recv_correlation_id, = struct.unpack_from('>i', resp, 0)
        if self._api_version == (0, 8, 2) and resp_type is GroupCoordinatorResponse and (correlation_id != 0) and (recv_correlation_id == 0):
            log.warning('Kafka 0.8.2 quirk -- GroupCoordinatorResponse coorelation id does not match request. This should go away once at least one topic has been initialized on the broker')
        elif correlation_id != recv_correlation_id:
            error = Errors.CorrelationIdError(f'Correlation ids do not match: sent {correlation_id}, recv {recv_correlation_id}')
            if not fut.done():
                fut.set_exception(error)
            self.close(reason=CloseReason.OUT_OF_SYNC)
            return
        if not fut.done():
            response = resp_type.decode(resp[4:])
            log.debug('%s Response %d: %s', self, correlation_id, response)
            fut.set_result(response)
    self._last_action = time.monotonic()
    self._requests.popleft()