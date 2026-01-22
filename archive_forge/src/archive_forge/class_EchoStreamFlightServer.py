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
class EchoStreamFlightServer(EchoFlightServer):
    """An echo server that streams individual record batches."""

    def do_get(self, context, ticket):
        return flight.GeneratorStream(self.last_message.schema, self.last_message.to_batches(max_chunksize=1024))

    def list_actions(self, context):
        return []

    def do_action(self, context, action):
        if action.type == 'who-am-i':
            return [context.peer_identity(), context.peer().encode('utf-8')]
        raise NotImplementedError