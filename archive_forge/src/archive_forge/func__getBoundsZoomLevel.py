from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _getBoundsZoomLevel(lon_min, lon_max, lat_min, lat_max, mapDim):
    """
    Get the mapbox zoom level given bounds and a figure dimension
    Source: https://stackoverflow.com/questions/6048975/google-maps-v3-how-to-calculate-the-zoom-level-for-a-given-bounds
    """
    scale = 2
    WORLD_DIM = {'height': 256 * scale, 'width': 256 * scale}
    ZOOM_MAX = 18

    def latRad(lat):
        sin = np.sin(lat * np.pi / 180)
        radX2 = np.log((1 + sin) / (1 - sin)) / 2
        return max(min(radX2, np.pi), -np.pi) / 2

    def zoom(mapPx, worldPx, fraction):
        return 0.95 * np.log(mapPx / worldPx / fraction) / np.log(2)
    latFraction = (latRad(lat_max) - latRad(lat_min)) / np.pi
    lngDiff = lon_max - lon_min
    lngFraction = (lngDiff + 360 if lngDiff < 0 else lngDiff) / 360
    latZoom = zoom(mapDim['height'], WORLD_DIM['height'], latFraction)
    lngZoom = zoom(mapDim['width'], WORLD_DIM['width'], lngFraction)
    return min(latZoom, lngZoom, ZOOM_MAX)