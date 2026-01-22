import json
import logging
import os.path
import click
import cligj
import rasterio
from rasterio.rio import options
from rasterio.rio.helpers import write_features
from rasterio.warp import transform_bounds
class _Collection:
    """For use with `rasterio.rio.helpers.write_features()`."""

    def __init__(self, dataset, bidx, precision=6, geographic=True):
        """Export raster dataset windows to GeoJSON polygon features.

        Parameters
        ----------
        dataset : a dataset object opened in 'r' mode
            Source dataset
        bidx : int
            Extract windows from this band
        precision : int, optional
            Coordinate precision
        geographic : bool, optional
            Reproject geometries to ``EPSG:4326`` if ``True``

        Yields
        ------
        dict
            GeoJSON polygon feature
        """
        self._src = dataset
        self._bidx = bidx
        self._precision = precision
        self._geographic = geographic

    def _normalize_bounds(self, bounds):
        if self._geographic:
            bounds = transform_bounds(self._src.crs, 'EPSG:4326', *bounds)
        if self._precision >= 0:
            bounds = (round(v, self._precision) for v in bounds)
        return bounds

    @property
    def bbox(self):
        return tuple(self._normalize_bounds(self._src.bounds))

    def __call__(self):
        gen = self._src.block_windows(bidx=self._bidx)
        for idx, (block, window) in enumerate(gen):
            bounds = self._normalize_bounds(self._src.window_bounds(window))
            xmin, ymin, xmax, ymax = bounds
            yield {'type': 'Feature', 'id': '{0}:{1}'.format(os.path.basename(self._src.name), idx), 'properties': {'block': json.dumps(block), 'window': window.todict()}, 'geometry': {'type': 'Polygon', 'coordinates': [[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]]}}