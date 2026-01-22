import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
class FionaReader:
    """
    Provides an interface for accessing the contents of a shapefile
    with the fiona library, which has a much faster reader than PyShp. See
    `fiona.open
    <https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open>`_
    for additional information on supported kwargs.

    The primary methods used on a FionaReader instance are
    :meth:`~cartopy.io.shapereader.FionaReader.records` and
    :meth:`~cartopy.io.shapereader.FionaReader.geometries`.

    """

    def __init__(self, filename, bbox=None, **kwargs):
        self._data = []
        with fiona.open(filename, **kwargs) as f:
            if bbox is not None:
                assert len(bbox) == 4
                features = f.filter(bbox=bbox)
            else:
                features = f
            if hasattr(features, '__geo_interface__'):
                fs = features.__geo_interface__
            else:
                fs = features
            if isinstance(fs, dict) and fs.get('type') == 'FeatureCollection':
                features_lst = fs['features']
            else:
                features_lst = features
            for feature in features_lst:
                if hasattr(f, '__geo_interface__'):
                    feature = feature.__geo_interface__
                else:
                    feature = feature
                d = {'geometry': sgeom.shape(feature['geometry']) if feature['geometry'] else None}
                d.update(feature['properties'])
                self._data.append(d)

    def close(self):
        pass

    def __len__(self):
        return len(self._data)

    def geometries(self):
        """
        Returns an iterator of shapely geometries from the shapefile.

        This interface is useful for accessing the geometries of the
        shapefile where knowledge of the associated metadata is desired.
        In the case where further metadata is needed use the
        :meth:`~cartopy.io.shapereader.FionaReader.records`
        interface instead, extracting the geometry from the record with the
        :meth:`~cartopy.io.shapereader.FionaRecord.geometry` method.

        """
        for item in self._data:
            yield item['geometry']

    def records(self):
        """
        Returns an iterator of :class:`~FionaRecord` instances.

        """
        for item in self._data:
            yield FionaRecord(item['geometry'], {key: value for key, value in item.items() if key != 'geometry'})