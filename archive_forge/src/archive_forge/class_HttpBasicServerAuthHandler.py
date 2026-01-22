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
class HttpBasicServerAuthHandler(ServerAuthHandler):
    """An example implementation of HTTP basic authentication."""

    def __init__(self, creds):
        super().__init__()
        self.creds = creds

    def authenticate(self, outgoing, incoming):
        buf = incoming.read()
        auth = flight.BasicAuth.deserialize(buf)
        if auth.username not in self.creds:
            raise flight.FlightUnauthenticatedError('unknown user')
        if self.creds[auth.username] != auth.password:
            raise flight.FlightUnauthenticatedError('wrong password')
        outgoing.write(tobytes(auth.username))

    def is_valid(self, token):
        if not token:
            raise flight.FlightUnauthenticatedError('token not provided')
        if token not in self.creds:
            raise flight.FlightUnauthenticatedError('unknown user')
        return token