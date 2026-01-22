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
class HeaderAuthServerMiddlewareFactory(ServerMiddlewareFactory):
    """Validates incoming username and password."""

    def start_call(self, info, headers):
        auth_header = case_insensitive_header_lookup(headers, 'Authorization')
        values = auth_header[0].split(' ')
        token = ''
        error_message = 'Invalid credentials'
        if values[0] == 'Basic':
            decoded = base64.b64decode(values[1])
            pair = decoded.decode('utf-8').split(':')
            if not (pair[0] == 'test' and pair[1] == 'password'):
                raise flight.FlightUnauthenticatedError(error_message)
            token = 'token1234'
        elif values[0] == 'Bearer':
            token = values[1]
            if not token == 'token1234':
                raise flight.FlightUnauthenticatedError(error_message)
        else:
            raise flight.FlightUnauthenticatedError(error_message)
        return HeaderAuthServerMiddleware(token)