import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
class ErrorFlightServer(FlightServerBase):
    """A Flight server that uses all the Flight-specific errors."""

    @staticmethod
    def error_cases():
        return {'internal': flight.FlightInternalError, 'timedout': flight.FlightTimedOutError, 'cancel': flight.FlightCancelledError, 'unauthenticated': flight.FlightUnauthenticatedError, 'unauthorized': flight.FlightUnauthorizedError, 'notimplemented': NotImplementedError, 'invalid': pa.ArrowInvalid, 'key': KeyError}

    def do_action(self, context, action):
        error_cases = ErrorFlightServer.error_cases()
        if action.type in error_cases:
            raise error_cases[action.type]('foo')
        elif action.type == 'protobuf':
            err_msg = b'this is an error message'
            raise flight.FlightUnauthorizedError('foo', err_msg)
        raise NotImplementedError

    def list_flights(self, context, criteria):
        yield flight.FlightInfo(pa.schema([]), flight.FlightDescriptor.for_path('/foo'), [], -1, -1)
        raise flight.FlightInternalError('foo')

    def do_put(self, context, descriptor, reader, writer):
        if descriptor.command == b'internal':
            raise flight.FlightInternalError('foo')
        elif descriptor.command == b'timedout':
            raise flight.FlightTimedOutError('foo')
        elif descriptor.command == b'cancel':
            raise flight.FlightCancelledError('foo')
        elif descriptor.command == b'unauthenticated':
            raise flight.FlightUnauthenticatedError('foo')
        elif descriptor.command == b'unauthorized':
            raise flight.FlightUnauthorizedError('foo')
        elif descriptor.command == b'protobuf':
            err_msg = b'this is an error message'
            raise flight.FlightUnauthorizedError('foo', err_msg)