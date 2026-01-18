from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
def to_optional_bytes(value, errors='strict'):
    """Return the given value as bytes encoded using UTF-8 if not already bytes, or None if the value is None."""
    return None if value is None else to_bytes(value, errors)