from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import param
from bokeh.models import CustomJS
from pyviz_comms import JupyterComm
from ..util import lazy_load
from ..viewable import Viewable
from .base import ModelPane

        Register a Javascript event handler which triggers when the
        specified event is triggered. The callback can be a snippet
        of Javascript code or a bokeh CustomJS object making it possible
        to manipulate other models in response to an event.

        Reference: https://apache.github.io/echarts-handbook/en/concepts/event/

        Arguments
        ---------
        event: str
            The name of the event to register a handler on, e.g. 'click'.
        code: str
            The event handler to be executed when the event fires.
        query: str | None
            A query that determines when the event fires.
        args: Viewable
            A dictionary of Viewables to make available in the namespace
            of the object.
        