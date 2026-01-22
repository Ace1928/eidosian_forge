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
class HeadersTrailersMiddlewareFactory(ClientMiddlewareFactory):

    def __init__(self):
        self.headers = []

    def start_call(self, info):
        return HeadersTrailersMiddleware(self)