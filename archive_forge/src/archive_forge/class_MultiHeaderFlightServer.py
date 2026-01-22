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
class MultiHeaderFlightServer(FlightServerBase):
    """Test sending/receiving multiple (binary-valued) headers."""

    def do_action(self, context, action):
        middleware = context.get_middleware('test')
        headers = repr(middleware.client_headers).encode('utf-8')
        return [headers]