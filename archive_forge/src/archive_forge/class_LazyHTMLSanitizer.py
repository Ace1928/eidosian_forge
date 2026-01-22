from __future__ import annotations
import ast
import base64
import datetime as dt
import json
import logging
import numbers
import os
import pathlib
import re
import sys
import urllib.parse as urlparse
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from functools import partial
from html import escape  # noqa
from importlib import import_module
from typing import Any, AnyStr
import bokeh
import numpy as np
import param
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from packaging.version import Version
from .checks import (  # noqa
from .parameters import (  # noqa
class LazyHTMLSanitizer:
    """
    Wraps bleach.sanitizer.Cleaner lazily importing it on the first
    call to the clean method.
    """

    def __init__(self, **kwargs):
        self._cleaner = None
        self._kwargs = kwargs

    def clean(self, text):
        if self._cleaner is None:
            import bleach
            self._cleaner = bleach.sanitizer.Cleaner(**self._kwargs)
        return self._cleaner.clean(text)