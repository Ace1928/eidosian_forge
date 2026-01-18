import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def write_binary_file(self, path, data):
    self.ensure_dir(os.path.dirname(path))
    if not self.dry_run:
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as f:
            f.write(data)
    self.record_as_written(path)