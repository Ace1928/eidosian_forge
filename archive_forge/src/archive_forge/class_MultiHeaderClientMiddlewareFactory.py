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
class MultiHeaderClientMiddlewareFactory(ClientMiddlewareFactory):
    """Test sending/receiving multiple (binary-valued) headers."""

    def __init__(self):
        self.last_headers = {}

    def start_call(self, info):
        return MultiHeaderClientMiddleware(self)