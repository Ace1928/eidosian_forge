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
class SelectiveAuthServerMiddlewareFactory(ServerMiddlewareFactory):
    """Deny access to certain methods based on a header."""

    def start_call(self, info, headers):
        if info.method == flight.FlightMethod.LIST_ACTIONS:
            return
        token = headers.get('x-auth-token')
        if not token:
            raise flight.FlightUnauthenticatedError('No token')
        token = token[0]
        if token != 'password':
            raise flight.FlightUnauthenticatedError('Invalid token')
        return HeaderServerMiddleware(token)