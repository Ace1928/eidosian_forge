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
def test_timeout_passes():
    """Make sure timeouts do not fire on fast requests."""
    with ConstantFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        options = flight.FlightCallOptions(timeout=5.0)
        client.do_get(flight.Ticket(b'ints'), options=options).read_all()