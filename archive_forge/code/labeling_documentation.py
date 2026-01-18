from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
 Select labels based on a user-defined policy function.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing it to Bokeh.

    