from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
class WMTSTileSource(MercatorTileSource):
    """ Behaves much like ``TMSTileSource`` but has its tile-origin in the
    top-left.

    This is the most common used tile source for web mapping applications.
    Such companies as Google, MapQuest, Stadia, Esri, and OpenStreetMap provide
    service which use the WMTS specification e.g. ``http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)