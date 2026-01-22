import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def _locations_tolist(x):
    """Transforms recursively a list of iterables into a list of list."""
    if hasattr(x, '__iter__'):
        return list(map(_locations_tolist, x))
    else:
        return x