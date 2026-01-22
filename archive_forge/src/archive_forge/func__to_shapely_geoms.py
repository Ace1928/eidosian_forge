import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _to_shapely_geoms(self, response):
    """
        Convert polygon coordinate strings in WFS response XML to Shapely
        geometries.

        Parameters
        ----------
        response: (file-like object)
            WFS response XML data.

        Returns
        -------
        geoms_by_srs
            A dictionary containing geometries, with key-value pairs of
            the form {srsname: [geoms]}.

        """
    linear_rings_data = []
    linestrings_data = []
    points_data = []
    tree = ElementTree.parse(response)
    for node in tree.iter():
        snode = str(node)
        if _MAP_SERVER_NS in snode or (self.url and self.url in snode):
            s1 = snode.split()[1]
            tag = s1[s1.find('}') + 1:-1]
            if 'geom' in tag or 'Geom' in tag:
                find_str = f'.//{_GML_NS}LinearRing'
                if self._node_has_child(node, find_str):
                    data = self._find_polygon_coords(node, find_str)
                    linear_rings_data.extend(data)
                find_str = f'.//{_GML_NS}LineString'
                if self._node_has_child(node, find_str):
                    data = self._find_polygon_coords(node, find_str)
                    linestrings_data.extend(data)
                find_str = f'.//{_GML_NS}Point'
                if self._node_has_child(node, find_str):
                    data = self._find_polygon_coords(node, find_str)
                    points_data.extend(data)
    geoms_by_srs = {}
    for srs, x, y in linear_rings_data:
        geoms_by_srs.setdefault(srs, []).append(sgeom.LinearRing(zip(x, y)))
    for srs, x, y in linestrings_data:
        geoms_by_srs.setdefault(srs, []).append(sgeom.LineString(zip(x, y)))
    for srs, x, y in points_data:
        geoms_by_srs.setdefault(srs, []).append(sgeom.Point(zip(x, y)))
    return geoms_by_srs