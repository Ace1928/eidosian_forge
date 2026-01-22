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
def _load_pypirc(index):
    """
    Read the PyPI access configuration as supported by distutils.
    """
    return PyPIRCFile(url=index.url).read()