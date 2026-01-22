import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def StopCapturing():
    while _captured_streams:
        _, cap_stream = _captured_streams.popitem()
        cap_stream.StopCapture()
        del cap_stream