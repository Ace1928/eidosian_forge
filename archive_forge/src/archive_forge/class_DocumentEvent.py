from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class DocumentEvent(Event):
    """ Base class for all Bokeh Document events.

    This base class is not typically useful to instantiate on its own.

    """