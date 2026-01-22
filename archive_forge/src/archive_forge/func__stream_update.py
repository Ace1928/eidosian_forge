from collections import defaultdict
from itertools import groupby
import numpy as np
import param
from bokeh.layouts import gridplot
from bokeh.models import (
from bokeh.models.layouts import TabPanel, Tabs
from ...core import (
from ...core.options import SkipRendering
from ...core.util import (
from ...selection import NoOpSelectionDisplay
from ..links import Link
from ..plot import (
from ..util import attach_streams, collate, displayable
from .links import LinkCallback
from .util import (
def _stream_update(self, **kwargs):
    contents = [k for s in self.streams for k in s.contents]
    key = tuple((None if d in contents else k for d, k in zip(self.dimensions, self.current_key)))
    key = wrap_tuple_streams(key, self.dimensions, self.streams)
    self._get_title_div(key)