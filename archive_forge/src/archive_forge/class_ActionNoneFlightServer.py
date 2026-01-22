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
class ActionNoneFlightServer(EchoFlightServer):
    """A server that implements a side effect to a non iterable action."""
    VALUES = []

    def do_action(self, context, action):
        if action.type == 'get_value':
            return [json.dumps(self.VALUES).encode('utf-8')]
        elif action.type == 'append':
            self.VALUES.append(True)
            return None
        raise NotImplementedError