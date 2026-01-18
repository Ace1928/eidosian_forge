from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
 Has the same default tile origin as the ``WMTSTileSource`` but requested
    tiles use a ``{XMIN}``, ``{YMIN}``, ``{XMAX}``, ``{YMAX}`` e.g.
    ``http://your.custom.tile.service?bbox={XMIN},{YMIN},{XMAX},{YMAX}``.

    