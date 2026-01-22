import base64
import json
import warnings
from binascii import hexlify
from collections import OrderedDict
from html import escape
from os import urandom
from pathlib import Path
from urllib.request import urlopen
from jinja2 import Environment, PackageLoader, Template
from .utilities import _camelify, _parse_size, none_max, none_min
def _get_self_bounds(self):
    """Computes the bounds of the object itself (not including it's children)
        in the form [[lat_min, lon_min], [lat_max, lon_max]]
        """
    return [[None, None], [None, None]]