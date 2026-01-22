from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
class StreamlitInvalidColorError(StreamlitAPIException):

    def __init__(self, df, color_from_user, *args):
        ', '.join((str(c) for c in list(df.columns)))
        message = f'\nThis does not look like a valid color argument: `{color_from_user}`.\n\nThe color argument can be:\n\n* A hex string like "#ffaa00" or "#ffaa0088".\n* An RGB or RGBA tuple with the red, green, blue, and alpha\n  components specified as ints from 0 to 255 or floats from 0.0 to\n  1.0.\n* The name of a column.\n* Or a list of colors, matching the number of y columns to draw.\n        '
        super().__init__(message, *args)