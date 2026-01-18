from __future__ import annotations
import logging # isort:skip
import sys
from typing import Any, cast
from ..settings import settings

    A logging.basicConfig() wrapper that also undoes the default
    Bokeh-specific configuration.
    