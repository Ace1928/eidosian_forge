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
class TracingFlightServer(FlightServerBase):
    """A server that echoes back trace context values."""

    def do_action(self, context, action):
        trace_context = context.get_middleware('tracing').trace_context
        return (f'{key}: {value}'.encode('utf-8') for key, value in trace_context.items())